"""Tests for the seed ensemble (run: pytest tests -v)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("keras")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecast import data_loader as dl  # noqa: E402
from src.forecast import ensemble as en  # noqa: E402
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


@pytest.fixture
def cfg():
    base = dl.load_config()
    return {**base, "forecast": {"window_size": 12, "horizon": 4, "target_mode": "delta"},
            "split": {"val_weeks": 26, "test_weeks": 26},
            "model": {**base["model"], "epochs": 2, "batch_size": 16, "lstm_units": [8, 4], "dense_units": 8,
                      "use_seasonal_input": True}}


def test_seed_validation(cfg):
    assert en.get_ensemble_seeds(cfg) == en.DEFAULT_SEEDS
    assert en.get_ensemble_seeds(cfg, [1, 2]) == [1, 2]
    with pytest.raises(ValueError):
        en.get_ensemble_seeds(cfg, [1])
    with pytest.raises(ValueError):
        en.get_ensemble_seeds(cfg, [3, 3])


def test_average_and_blend():
    a, b = np.ones((2, 4)), 3 * np.ones((2, 4))
    assert np.allclose(en.average_forecasts([a, b]), 2.0)
    assert np.allclose(en.blend(a, b, 0.25), 2.5)
    with pytest.raises(ValueError):
        en.average_forecasts([a, np.ones((3, 4))])
    with pytest.raises(ValueError):
        en.blend(a, b, 1.5)


def test_ensemble_end_to_end_val_only(cfg, tmp_path):
    weekly = synthetic_weekly()
    sg.prepare_dataset(cfg, weekly=weekly, holidays=synthetic_holidays(), out_root=tmp_path, save=True)
    res = en.run_ensemble(cfg, seeds=[1, 2], root=tmp_path, weekly=weekly)
    assert set(res["overall"]) == {"lstm", "lstm_single", "naive", "moving_average", "seasonal_naive", en.BLEND_NAME}
    assert len(res["members"]) == 2 and res["members"]["seed"].tolist() == [1, 2]
    assert np.isfinite(res["overall"]["lstm"]["WAPE"])
    meta = json.loads((tmp_path / "models" / "ensemble_meta.json").read_text())
    assert meta["seeds"] == [1, 2] and len(meta["files"]) == 2
    assert (tmp_path / "models" / "lstm_demand_model_ens_s1.keras").exists()
    assert not (tmp_path / "models" / "lstm_demand_model.keras").exists()  # main model untouched
    assert (tmp_path / "outputs" / "ensemble_val.json").exists()
    assert not (tmp_path / "outputs" / "evaluation_test.json").exists()


def test_ensemble_refuses_test_split(cfg):
    with pytest.raises(ValueError):
        en.evaluate_ensemble(cfg, {}, [], pd.DataFrame(), split="test")
