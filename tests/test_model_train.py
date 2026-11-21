"""Unit tests for model.py and train.py (run: pytest tests -v). Skipped if Keras is not installed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("keras")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecast import data_loader as dl  # noqa: E402
from src.forecast import model as md  # noqa: E402
from src.forecast import sequence_generator as sg  # noqa: E402
from src.forecast import train as tr  # noqa: E402

WINDOW, N_FEAT, HORIZON, N_PROD = 12, 18, 4, 3


@pytest.fixture
def config():
    return dl.load_config()


def random_data(n=48, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, WINDOW, N_FEAT)).astype(np.float32)
    meta = pd.DataFrame({"product_idx": rng.integers(0, N_PROD, n)})
    return X, meta


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


def test_get_model_params_defaults_and_validation(config):
    assert md.get_model_params({})["lstm_units"] == [64, 32]
    with pytest.raises(ValueError):
        md.get_model_params({"model": {"lstm_units": []}})
    with pytest.raises(ValueError):
        md.get_model_params({"model": {"dropout": 1.5}})
    with pytest.raises(ValueError):
        md.get_model_params({"model": {"loss": "unknown"}})


@pytest.mark.parametrize("use_emb", [True, False])
def test_output_shape_with_and_without_embedding(use_emb):
    params = {**md.MODEL_DEFAULTS, "use_product_embedding": use_emb}
    model = md.build_lstm_model(WINDOW, N_FEAT, HORIZON, N_PROD, params)
    X, meta = random_data()
    out = model.predict(md.make_inputs(X, meta, md.uses_embedding(model)), verbose=0)
    assert out.shape == (len(X), HORIZON) and np.isfinite(out).all()
    assert md.uses_embedding(model) == use_emb


def test_architecture_follows_config():
    params = {**md.MODEL_DEFAULTS, "lstm_units": [16, 8, 4], "dense_units": 0}
    model = md.build_lstm_model(WINDOW, N_FEAT, HORIZON, N_PROD, params)
    names = [layer.name for layer in model.layers]
    assert sum(n.startswith("lstm_") for n in names) == 3
    assert "dense_hidden" not in names and "forecast" in names


def test_make_inputs_formats():
    X, meta = random_data(5)
    assert md.make_inputs(X, meta, False) is X
    d = md.make_inputs(X, meta, True)
    assert d["sequence"].shape == (5, WINDOW, N_FEAT) and d["product"].shape == (5, 1) and d["product"].dtype == np.int32


def test_same_seed_gives_same_initial_model():
    X, meta = random_data(8)
    outs = []
    for _ in range(2):
        md.set_seed(123)
        m = md.build_lstm_model(WINDOW, N_FEAT, HORIZON, N_PROD, md.MODEL_DEFAULTS)
        outs.append(m.predict(md.make_inputs(X, meta, True), verbose=0))
    assert np.allclose(outs[0], outs[1])


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        md.load_model(tmp_path / "nope.keras")


def test_train_end_to_end_saves_and_reloads(config, tmp_path):
    cfg = {**config, "forecast": {"window_size": 12, "horizon": 4}, "split": {"val_weeks": 26, "test_weeks": 26},
           "model": {**config["model"], "epochs": 3, "batch_size": 16, "lstm_units": [8, 4], "dense_units": 8}}
    sg.prepare_dataset(cfg, weekly=synthetic_weekly(), holidays=synthetic_holidays(), out_root=tmp_path, save=True)
    res = tr.train_model(cfg, root=tmp_path, verbose=0)

    for f in ("models/lstm_demand_model.keras", "outputs/training_history.csv",
              "outputs/training_summary.json", "outputs/plots/06_training_loss.png"):
        assert (tmp_path / f).exists(), f
    assert res["summary"]["epochs_run"] >= 1 and np.isfinite(res["summary"]["best_val_loss"])

    # the saved model reproduces the predictions of the in-memory (best-weights) model
    prepared = sg.load_prepared(cfg, root=tmp_path)
    val = prepared["sequences"]["val"]
    reloaded = md.load_model(tmp_path / "models" / "lstm_demand_model.keras")
    p_saved = reloaded.predict(md.model_inputs(reloaded, val), verbose=0)
    p_live = res["model"].predict(md.model_inputs(res["model"], val), verbose=0)
    assert p_saved.shape == (len(val["X"]), 4)
    assert np.allclose(p_saved, p_live, atol=1e-4)


def test_train_raises_on_empty_validation(config, tmp_path):
    cfg = {**config, "model": {**config["model"], "epochs": 1}}
    empty = {"sequences": {"train": {"X": np.zeros((0, 12, 18), np.float32), "y": np.zeros((0, 4), np.float32), "meta": pd.DataFrame()},
                           "val": {"X": np.zeros((0, 12, 18), np.float32), "y": np.zeros((0, 4), np.float32), "meta": pd.DataFrame()}},
             "info": {"products": ["A"], "horizon": 4}}
    with pytest.raises(ValueError):
        tr.train_model(cfg, prepared=empty, root=tmp_path, verbose=0)
