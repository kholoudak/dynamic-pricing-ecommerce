"""Tests for the prediction function (run: pytest tests -v)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("keras")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecast import data_loader as dl  # noqa: E402
from src.forecast import ensemble as en  # noqa: E402
from src.forecast import predict as pr  # noqa: E402
from src.forecast import sequence_generator as sg  # noqa: E402


def synthetic_weekly(n_weeks=140, products=("A", "B", "C"), seed=0):
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


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    base = dl.load_config()
    cfg = {**base, "forecast": {"window_size": 12, "horizon": 4, "target_mode": "delta", "season_length": 52},
           "split": {"val_weeks": 26, "test_weeks": 26},
           "model": {**base["model"], "epochs": 2, "batch_size": 16, "lstm_units": [8, 4], "dense_units": 8,
                     "use_seasonal_input": True}}
    weekly = synthetic_weekly()
    (root / "outputs").mkdir()
    weekly.to_csv(root / "outputs" / "sales_weekly.csv", index=False)
    sg.prepare_dataset(cfg, weekly=weekly, holidays=synthetic_holidays(), out_root=root, save=True)
    en.run_ensemble(cfg, seeds=[1, 2], root=root, weekly=weekly)
    return cfg, root, weekly


def test_prediction_params_validation():
    assert pr.get_prediction_params({}) == {"use_ensemble": True, "seasonal_weight": 0.5}
    with pytest.raises(ValueError):
        pr.get_prediction_params({"prediction": {"seasonal_weight": 1.5}})


def test_to_week_start():
    assert pr.to_week_start("2017-08-16") == pd.Timestamp("2017-08-14")  # Wednesday -> Monday
    assert pr.to_week_start("2017-08-14") == pd.Timestamp("2017-08-14")


def test_forecast_table_shape_and_range(project):
    cfg, root, weekly = project
    f = pr.DemandForecaster(cfg, root)
    t = f.forecast_table()
    assert len(t) == 3 * 4 and set(t["horizon"]) == {1, 2, 3, 4}
    assert np.isfinite(t["demand_pred"]).all() and (t["demand_pred"] >= 0).all()
    assert f.origin_week == weekly["week"].max()
    assert t["week"].min() == weekly["week"].max() + pd.Timedelta(days=7)
    # default weight 0.5: the final value is the mean of the two components
    assert np.allclose(t["demand_pred"], 0.5 * t["lstm_pred"] + 0.5 * t["seasonal_naive_pred"])


def test_seasonal_component_is_last_years_sales(project):
    cfg, root, weekly = project
    t = pr.DemandForecaster(cfg, root).forecast_table()
    row = t[(t["product_id"] == "B") & (t["horizon"] == 2)].iloc[0]
    last_year = row["week"] - pd.Timedelta(days=7 * 52)
    expected = weekly[(weekly["product_id"] == "B") & (weekly["week"] == last_year)]["sales"].iloc[0]
    assert row["seasonal_naive_pred"] == pytest.approx(expected)


def test_weight_zero_is_pure_lstm_and_weight_one_is_seasonal(project):
    cfg, root, _ = project
    t0 = pr.DemandForecaster({**cfg, "prediction": {"seasonal_weight": 0.0}}, root).forecast_table()
    assert np.allclose(t0["demand_pred"], t0["lstm_pred"].clip(lower=0))
    t1 = pr.DemandForecaster({**cfg, "prediction": {"seasonal_weight": 1.0}}, root).forecast_table()
    assert np.allclose(t1["demand_pred"], t1["seasonal_naive_pred"])


def test_predict_by_horizon_and_by_date_agree(project):
    cfg, root, weekly = project
    f = pr.DemandForecaster(cfg, root)
    target = weekly["week"].max() + pd.Timedelta(days=14)
    a = f.predict("A", 2)
    b = f.predict("A", (target + pd.Timedelta(days=3)).date().isoformat())  # a Thursday of that week
    assert a == pytest.approx(b)


def test_invalid_inputs(project):
    cfg, root, weekly = project
    f = pr.DemandForecaster(cfg, root)
    with pytest.raises(ValueError):
        f.predict("NOPE", 1)
    for bad in (0, 5, -1, True):
        with pytest.raises(ValueError):
            f.predict("A", bad)
    with pytest.raises(ValueError):
        f.predict("A", "2010-01-04")  # past
    with pytest.raises(ValueError):
        f.predict("A", "not a date")


def test_inputs_use_only_the_last_window_and_detect_stale_series(project):
    cfg, root, _ = project
    feats = pd.read_csv(root / "data" / "processed" / "features_weekly.csv", parse_dates=["week"])
    scaler = sg.FeatureScaler.load(root / "models" / "feature_scaler.json")
    cols = scaler.feature_cols
    prod = {"A": 0, "B": 1, "C": 2}
    inp = pr.build_inference_inputs(feats, scaler, cols, 12, 4, prod)
    assert inp["X"].shape == (3, 12, len(cols)) and inp["S"].shape == (3, 5)
    stale = feats[~((feats["product_id"] == "C") & (feats["week"] == feats["week"].max()))]
    with pytest.raises(ValueError):
        pr.build_inference_inputs(stale, scaler, cols, 12, 4, prod)


def test_single_model_mode_requires_main_model(project, tmp_path):
    cfg, root, _ = project
    with pytest.raises(FileNotFoundError):  # main model was never trained in this project
        pr.DemandForecaster({**cfg, "prediction": {"use_ensemble": False}}, root)
