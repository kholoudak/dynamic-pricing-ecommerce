"""Unit tests for src/forecast/evaluate.py (run: pytest tests -v)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecast import data_loader as dl  # noqa: E402
from src.forecast import evaluate as ev  # noqa: E402
from src.forecast import sequence_generator as sg  # noqa: E402


def synthetic_weekly(n_weeks=120, products=("A", "B", "C"), seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2014-01-06", periods=n_weeks, freq="7D")
    rows = []
    for k, p in enumerate(products):
        base = 100 * (k + 1)
        for i, w in enumerate(weeks):
            rows.append((p, w, base * (1 + 0.3 * np.sin(2 * np.pi * i / 52)) + rng.normal(0, 3),
                         0 if i < 10 else int(rng.integers(1, 30)), 50 + (i % 3), False))
    return pd.DataFrame(rows, columns=["product_id", "week", "sales", "onpromotion", "open_stores", "is_outlier"])


def synthetic_holidays():
    return pd.DataFrame({"date": pd.to_datetime(["2014-01-01"]), "type": ["Holiday"], "locale": ["National"],
                         "locale_name": ["Ecuador"], "description": ["x"], "transferred": [False]})


def small_lookup():
    """Product A: sales = week index (0..59); product B: 100 + index; both start 2015-01-05."""
    weeks = pd.date_range("2015-01-05", periods=60, freq="7D")
    return pd.DataFrame({"A": np.arange(60, dtype=float), "B": 100 + np.arange(60, dtype=float)}, index=weeks)


def make_meta(lookup, origin_idx, product="A"):
    origin = lookup.index[origin_idx]
    return pd.DataFrame({"product_id": [product], "origin_week": [origin]})


# ------------------------------------------------------------------ metrics
def test_regression_metrics_known_values():
    r = ev.regression_metrics([100, 200, 0], [110, 180, 5])
    assert r["MAE"] == pytest.approx(35 / 3)
    assert r["RMSE"] == pytest.approx(np.sqrt(525 / 3))
    assert r["MAPE"] == pytest.approx(10.0)            # zero actual excluded
    assert r["MAPE_coverage_pct"] == pytest.approx(200 / 3)
    assert r["WAPE"] == pytest.approx(35 / 300 * 100)
    assert r["bias"] == pytest.approx((10 - 20 + 5) / 3)


def test_regression_metrics_perfect_and_invalid():
    r = ev.regression_metrics([1, 2, 3], [1, 2, 3])
    assert r["MAE"] == 0 and r["RMSE"] == 0 and r["WAPE"] == 0 and r["sMAPE"] == 0
    with pytest.raises(ValueError):
        ev.regression_metrics([1, 2], [1])


def test_wape_nan_when_all_actuals_are_zero():
    assert np.isnan(ev.regression_metrics([0, 0], [1, 1])["WAPE"])


# ------------------------------------------------------------------ baselines
def test_actual_units_and_baselines_on_known_series():
    lk = small_lookup()
    meta = make_meta(lk, 30)  # origin week index 30
    assert np.allclose(ev.actual_units(lk, meta, 4), [[31, 32, 33, 34]])
    b = ev.baseline_forecasts(lk, meta, 4, ma_window=4, season_length=52)
    assert np.allclose(b["naive"], [[30] * 4])
    assert np.allclose(b["moving_average"], [[np.mean([27, 28, 29, 30])] * 4])
    # target index 31..34 minus 52 weeks < 0 -> no data -> fall back to naive
    assert np.allclose(b["seasonal_naive"], [[30] * 4]) and b["seasonal_fallback_pct"] == 100


def test_seasonal_naive_uses_same_week_last_year():
    lk = small_lookup()
    meta = make_meta(lk, 55)  # targets 56..59 -> minus 52 = 4..7
    b = ev.baseline_forecasts(lk, meta, 4, season_length=52)
    assert np.allclose(b["seasonal_naive"], [[4, 5, 6, 7]]) and b["seasonal_fallback_pct"] == 0


def test_baselines_never_use_the_future():
    lk = small_lookup()
    meta = make_meta(lk, 40)
    before = ev.baseline_forecasts(lk, meta, 4)
    changed = lk.copy()
    changed.iloc[41:, :] = 1e9  # destroy everything after the origin
    after = ev.baseline_forecasts(changed, meta, 4)
    for name in ev.BASELINES:
        assert np.allclose(before[name], after[name]), name


def test_moving_average_with_short_history_uses_available_weeks():
    lk = small_lookup()
    meta = make_meta(lk, 1)  # only 2 weeks of history
    b = ev.baseline_forecasts(lk, meta, 4, ma_window=4)
    assert np.allclose(b["moving_average"], [[0.5] * 4])


def test_actual_units_raises_when_target_outside_table():
    lk = small_lookup()
    with pytest.raises(ValueError):
        ev.actual_units(lk, make_meta(lk, 58), 4)


# ------------------------------------------------------------------ comparison
def test_evaluate_models_structure_and_ranking():
    rng = np.random.default_rng(0)
    y = rng.uniform(50, 150, size=(20, 4))
    meta = pd.DataFrame({"product_id": ["A"] * 10 + ["B"] * 10})
    preds = {"lstm": y * 1.05, "naive": y * 1.5, "moving_average": y * 1.3, "seasonal_naive": y * 0.7}
    res = ev.evaluate_models(y, preds, meta)
    assert res["overall"]["lstm"]["WAPE"] == pytest.approx(5.0)
    assert res["n_products"] == 2 and len(res["by_horizon"]["lstm"]) == 4
    assert (res["by_product"]["best_model"] == "lstm").all()
    assert res["products_beaten"] == {"naive": 2, "moving_average": 2, "seasonal_naive": 2}
    assert res["macro_wape"]["lstm"] < res["macro_wape"]["naive"]
    assert any("Best model by pooled WAPE: lstm" in line for line in ev.summarize(res))


# ------------------------------------------------------------------ pipeline guards
def test_test_split_requires_final_flag():
    with pytest.raises(ValueError, match="FINAL"):
        ev.run_evaluation(split="test", final=False)
    with pytest.raises(ValueError):
        ev.run_evaluation(split="unknown")


def test_scaled_targets_match_weekly_table_after_inverse():
    cfg = {**dl.load_config(), "forecast": {"window_size": 12, "horizon": 4}, "split": {"val_weeks": 26, "test_weeks": 26}}
    weekly = synthetic_weekly()
    res = sg.prepare_dataset(cfg, weekly=weekly, holidays=synthetic_holidays(), save=False)
    lk = ev.build_sales_lookup(weekly)
    for split in sg.SPLITS:
        seq = res["sequences"][split]
        y_true = ev.actual_units(lk, seq["meta"], 4)
        back = res["scaler"].inverse_sales(seq["meta"]["product_id"].to_numpy(), seq["y"])
        assert np.allclose(back, y_true, rtol=1e-3, atol=1e-2)


def test_end_to_end_evaluation_writes_reports(tmp_path):
    pytest.importorskip("keras")
    from src.forecast import train as tr

    base = dl.load_config()
    cfg = {**base, "forecast": {"window_size": 12, "horizon": 4}, "split": {"val_weeks": 26, "test_weeks": 26},
           "model": {**base["model"], "epochs": 2, "batch_size": 16, "lstm_units": [8, 4], "dense_units": 8}}
    weekly = synthetic_weekly()
    sg.prepare_dataset(cfg, weekly=weekly, holidays=synthetic_holidays(), out_root=tmp_path, save=True)
    tr.train_model(cfg, root=tmp_path, verbose=0)

    res = ev.run_evaluation(cfg, split="val", root=tmp_path, weekly=weekly)
    for f in ("evaluation_val.json", "evaluation_val_by_product.csv", "predictions_val.csv",
              "plots/07_model_comparison_val.png", "plots/07_forecast_by_product_val.png"):
        assert (tmp_path / "outputs" / f).exists(), f
    report = json.loads((tmp_path / "outputs" / "evaluation_val.json").read_text())
    assert set(report["overall"]) == {"lstm", "naive", "moving_average", "seasonal_naive"}
    pred = pd.read_csv(tmp_path / "outputs" / "predictions_val.csv")
    assert set(pred["horizon"]) == {1, 2, 3, 4} and (pred["lstm"] >= 0).all()
    assert np.isfinite(res["overall"]["lstm"]["MAE"])

    # the test split works only with final=True
    res_test = ev.run_evaluation(cfg, split="test", final=True, root=tmp_path, weekly=weekly)
    assert res_test["n_products"] == 3
