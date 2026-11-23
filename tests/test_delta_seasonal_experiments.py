"""Tests for the delta target, the seasonal input and the experiments (run: pytest tests -v)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("keras")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecast import data_loader as dl  # noqa: E402
from src.forecast import evaluate as ev  # noqa: E402
from src.forecast import experiments as ex  # noqa: E402
from src.forecast import model as md  # noqa: E402
from src.forecast import sequence_generator as sg  # noqa: E402
from src.forecast import train as tr  # noqa: E402


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


@pytest.fixture
def cfg():
    base = dl.load_config()
    return {**base, "forecast": {"window_size": 12, "horizon": 4}, "split": {"val_weeks": 26, "test_weeks": 26},
            "model": {**base["model"], "epochs": 2, "batch_size": 16, "lstm_units": [8, 4], "dense_units": 8}}


# ------------------------------------------------------------------ seasonal features
def test_seasonal_features_values_and_flag():
    weeks = pd.date_range("2015-01-05", periods=80, freq="7D")
    zmap = {w: float(i) ** 1.0 * 0.5 for i, w in enumerate(weeks)}  # z_i = 0.5 * i
    out = sg.seasonal_features(zmap, weeks[60], horizon=4)
    # last year = weeks[8]; s_h = z[8+h] - z[8] = 0.5 * h
    assert np.allclose(out, [0.5, 1.0, 1.5, 2.0, 1.0])


def test_seasonal_features_zero_without_history():
    weeks = pd.date_range("2015-01-05", periods=80, freq="7D")
    zmap = {w: 1.0 for w in weeks}
    assert np.array_equal(sg.seasonal_features(zmap, weeks[20], 4), np.zeros(5, np.float32))


def test_seasonal_features_do_not_read_the_future():
    weeks = pd.date_range("2015-01-05", periods=80, freq="7D")
    zmap = {w: float(np.sin(i)) for i, w in enumerate(weeks)}
    origin = weeks[60]
    before = sg.seasonal_features(zmap, origin, 4)
    changed = {w: (v if w <= origin else 1e6) for w, v in zmap.items()}
    assert np.array_equal(before, sg.seasonal_features(changed, origin, 4))


def test_sequences_contain_seasonal_array(cfg):
    res = sg.prepare_dataset(cfg, weekly=synthetic_weekly(), holidays=synthetic_holidays(), save=False)
    for split in sg.SPLITS:
        S = res["sequences"][split]["S"]
        assert S.shape == (len(res["sequences"][split]["X"]), 5)
    assert (res["sequences"]["test"]["S"][:, -1] == 1).all()  # a year of history exists by the test period
    assert (res["sequences"]["train"]["S"][:, -1] == 0).any()  # but not at the very start of the training period


# ------------------------------------------------------------------ target encoding
def test_encode_decode_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 12, 18)).astype(np.float32)
    y = rng.normal(size=(10, 4)).astype(np.float32)
    assert md.encode_target(y, X, "level") is y
    d = md.encode_target(y, X, "delta", level_idx=0)
    assert np.allclose(d, y - X[:, -1, 0][:, None])
    assert np.allclose(md.decode_prediction(d, X, "delta", 0), y)


def test_target_mode_validation():
    assert md.get_target_mode({"forecast": {}}) == "level"
    assert md.get_target_mode({"forecast": {"target_mode": "delta"}}) == "delta"
    with pytest.raises(ValueError):
        md.get_target_mode({"forecast": {"target_mode": "other"}})


def test_seasonal_model_requires_S():
    params = {**md.MODEL_DEFAULTS, "use_seasonal_input": True}
    model = md.build_lstm_model(12, 18, 4, 3, params)
    assert md.uses_seasonal(model)
    seq = {"X": np.zeros((2, 12, 18), np.float32), "meta": pd.DataFrame({"product_idx": [0, 1]}), "S": None}
    with pytest.raises(ValueError):
        md.model_inputs(model, seq)
    seq["S"] = np.zeros((2, 5), np.float32)
    assert set(md.model_inputs(model, seq)) == {"sequence", "product", "seasonal"}


def test_delta_model_with_zero_output_equals_naive_forecast(cfg):
    """If the network outputs 0 in delta mode, the forecast must be exactly the naive baseline."""
    weekly = synthetic_weekly()
    prepared = sg.prepare_dataset(cfg, weekly=weekly, holidays=synthetic_holidays(), save=False)
    prepared["info"] = prepared["info"]
    model = md.build_lstm_model(12, len(prepared["feature_cols"]), 4, 3, md.MODEL_DEFAULTS)
    head = model.get_layer("forecast")
    head.set_weights([np.zeros_like(w) for w in head.get_weights()])
    seq = prepared["sequences"]["val"]
    meta_info = {"target_mode": "delta", "level_idx": 0}
    pred = ev.predict_lstm(model, seq, prepared["scaler"], meta_info)
    lookup = ev.build_sales_lookup(weekly)
    naive = ev.baseline_forecasts(lookup, seq["meta"], 4)["naive"]
    assert np.allclose(pred, naive, rtol=1e-3)


# ------------------------------------------------------------------ training / experiments
def test_train_delta_seasonal_saves_meta_and_evaluates(cfg, tmp_path):
    cfg = {**cfg, "forecast": {**cfg["forecast"], "target_mode": "delta"}, "model": {**cfg["model"], "use_seasonal_input": True}}
    weekly = synthetic_weekly()
    sg.prepare_dataset(cfg, weekly=weekly, holidays=synthetic_holidays(), out_root=tmp_path, save=True)
    res = tr.train_model(cfg, root=tmp_path, verbose=0)
    meta = json.loads((tmp_path / "models" / "model_meta.json").read_text())
    assert meta["target_mode"] == "delta" and meta["use_seasonal_input"] is True and meta["level_idx"] == 0
    assert res["summary"]["target_mode"] == "delta" and res["summary"]["uses_seasonal_input"]
    out = ev.run_evaluation(cfg, split="val", root=tmp_path, weekly=weekly)
    assert np.isfinite(out["overall"]["lstm"]["MAE"])


def test_load_model_meta_defaults_for_old_models(tmp_path):
    assert tr.load_model_meta(tmp_path / "missing.json") == {"target_mode": "level", "level_idx": 0, "use_seasonal_input": False}


def test_apply_variant_and_unknown_variant(cfg):
    v = ex.apply_variant(cfg, "delta_seasonal_reg")
    assert v["forecast"]["target_mode"] == "delta" and v["model"]["lstm_units"] == [32, 16] and v["model"]["dropout"] == 0.3
    assert cfg["forecast"].get("target_mode") is None  # the original config is untouched
    with pytest.raises(ValueError):
        ex.apply_variant(cfg, "nope")


def test_run_experiments_end_to_end(cfg, tmp_path):
    weekly = synthetic_weekly()
    sg.prepare_dataset(cfg, weekly=weekly, holidays=synthetic_holidays(), out_root=tmp_path, save=True)
    table = ex.run_experiments(cfg, root=tmp_path, variants=["delta"], include_level=True, weekly=weekly)
    assert set(table.index) == {"delta", "level", "baseline:naive", "baseline:moving_average", "baseline:seasonal_naive"}
    assert (tmp_path / "outputs" / "experiments_val.csv").exists()
    assert (tmp_path / "models" / "lstm_demand_model_exp_delta.keras").exists()
    assert not (tmp_path / "models" / "lstm_demand_model.keras").exists()  # main model untouched
    assert table["WAPE_pooled"].is_monotonic_increasing
