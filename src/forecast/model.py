"""
Step 6a - LSTM architecture / Etape 6a - Architecture du modele LSTM.

Input  : `sequence` (batch, window, n_features) = previous weeks, already scaled
         `product`  (batch, 1) = integer id of the product (optional embedding)
         `seasonal` (batch, horizon + 1) = last year's evolution over the same weeks (optional)
Output : (batch, horizon). Depending on forecast.target_mode:
         "level": scaled log-sales of the next weeks;
         "delta": CHANGE of scaled log-sales relative to the last observed week (output 0 = naive forecast).
         Use decode_prediction() then FeatureScaler.inverse_sales to go back to units.

Everything (layers, units, dropout, learning rate, loss) comes from config.yaml -> model.
Keras 3 on the TensorFlow backend, CPU only (a GPU is optional and never required).
"""
from __future__ import annotations

import os

# Set BEFORE importing TensorFlow: quieter logs, and identical numerical results between runs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path  # noqa: E402
from typing import Dict, Optional, Union  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import keras  # noqa: E402
from keras import layers  # noqa: E402

MODEL_DEFAULTS = {
    "lstm_units": [64, 32],
    "dropout": 0.2,
    "dense_units": 32,              # hidden Dense layer before the output (0 = none)
    "use_product_embedding": True,  # learn one small vector per product
    "embedding_dim": 4,
    "use_seasonal_input": False,    # extra input: what happened one year ago (see seasonal_features)
    "learning_rate": 0.001,
    "loss": "huber",                # "huber" (robust to spikes) or "mse"
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "model_file": "lstm_demand_model.keras",
}


def get_model_params(config: dict) -> dict:
    """Return the `model` settings of config.yaml, completed with defaults."""
    params = {**MODEL_DEFAULTS, **(config.get("model") or {})}
    units = params["lstm_units"]
    if not isinstance(units, (list, tuple)) or not units or any(int(u) <= 0 for u in units):
        raise ValueError(f"model.lstm_units must be a non-empty list of positive integers, got {units!r}")
    if not 0 <= float(params["dropout"]) < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if params["loss"] not in ("huber", "mse"):
        raise ValueError("model.loss must be 'huber' or 'mse'")
    return params


def set_seed(seed: int) -> None:
    """Fix Python, NumPy and Keras/TensorFlow random generators (reproducible runs)."""
    keras.utils.set_random_seed(int(seed))


TARGET_MODES = ("level", "delta")


def get_target_mode(config: dict) -> str:
    """Return forecast.target_mode ("level" by default) after validation."""
    mode = (config.get("forecast") or {}).get("target_mode", "level")
    if mode not in TARGET_MODES:
        raise ValueError(f"forecast.target_mode must be one of {TARGET_MODES}, got {mode!r}")
    return mode


def encode_target(y: np.ndarray, X: np.ndarray, mode: str, level_idx: int = 0) -> np.ndarray:
    """Training target: the scaled levels (mode "level") or their change vs the last input week ("delta").

    `X[:, -1, level_idx]` is the scaled log-sales of the origin week (the last week the model sees).
    """
    if mode == "level":
        return y
    return y - X[:, -1, level_idx][:, None]


def decode_prediction(pred: np.ndarray, X: np.ndarray, mode: str, level_idx: int = 0) -> np.ndarray:
    """Inverse of encode_target: network output -> scaled log-sales levels."""
    if mode == "level":
        return pred
    return pred + X[:, -1, level_idx][:, None]


def build_lstm_model(window: int, n_features: int, horizon: int, n_products: Optional[int] = None,
                     params: Optional[dict] = None) -> keras.Model:
    """Build and compile the configurable LSTM.

    Architecture: Input -> [LSTM -> Dropout] x len(lstm_units) -> (+ product embedding, + seasonal input)
                  -> Dense(dense_units, relu) -> Dropout -> Dense(horizon).
    Optimizer Adam; loss Huber or MSE; metric MAE (on scaled log-sales).
    """
    p = {**MODEL_DEFAULTS, **(params or {})}
    use_emb = bool(p["use_product_embedding"]) and n_products is not None
    use_seas = bool(p["use_seasonal_input"])

    seq_in = keras.Input(shape=(window, n_features), name="sequence")
    x = seq_in
    n_layers = len(p["lstm_units"])
    for i, units in enumerate(p["lstm_units"]):
        x = layers.LSTM(int(units), return_sequences=(i < n_layers - 1), name=f"lstm_{i + 1}")(x)
        x = layers.Dropout(float(p["dropout"]), name=f"dropout_{i + 1}")(x)

    inputs = [seq_in]
    if use_emb:
        prod_in = keras.Input(shape=(1,), dtype="int32", name="product")
        emb = layers.Embedding(int(n_products), int(p["embedding_dim"]), name="product_embedding")(prod_in)
        x = layers.Concatenate(name="concat")([x, layers.Flatten(name="flatten_emb")(emb)])
        inputs.append(prod_in)

    if use_seas:
        seas_in = keras.Input(shape=(int(horizon) + 1,), name="seasonal")
        x = layers.Concatenate(name="concat_seasonal")([x, seas_in])
        inputs.append(seas_in)

    if int(p["dense_units"]) > 0:
        x = layers.Dense(int(p["dense_units"]), activation="relu", name="dense_hidden")(x)
        x = layers.Dropout(float(p["dropout"]), name="dropout_head")(x)
    out = layers.Dense(int(horizon), name="forecast")(x)

    model = keras.Model(inputs=inputs if len(inputs) > 1 else seq_in, outputs=out, name="lstm_demand")
    loss = keras.losses.Huber(delta=1.0) if p["loss"] == "huber" else "mse"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(p["learning_rate"])),
                  loss=loss, metrics=["mae"])
    return model


def make_inputs(X: np.ndarray, meta: pd.DataFrame, use_embedding: bool, S: Optional[np.ndarray] = None,
                use_seasonal: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Arrange the arrays in the format expected by the model (dict when extra inputs are used)."""
    if not use_embedding and not use_seasonal:
        return X
    inputs = {"sequence": X}
    if use_embedding:
        inputs["product"] = meta["product_idx"].to_numpy(np.int32).reshape(-1, 1)
    if use_seasonal:
        if S is None:
            raise ValueError("This model needs the seasonal input S: regenerate the sequences (Step 5).")
        inputs["seasonal"] = S
    return inputs


def model_inputs(model: keras.Model, seq: dict):
    """make_inputs for one split dict {"X", "meta", "S"}, using what the model actually expects."""
    return make_inputs(seq["X"], seq["meta"], uses_embedding(model), seq.get("S"), uses_seasonal(model))


def uses_embedding(model: keras.Model) -> bool:
    """True if the model has the product input (works for freshly built and reloaded models)."""
    return any(i.name.startswith("product") for i in model.inputs)


def uses_seasonal(model: keras.Model) -> bool:
    """True if the model has the seasonal input."""
    return any(i.name.startswith("seasonal") for i in model.inputs)


def load_model(path: Path) -> keras.Model:
    """Load a saved .keras model (architecture + weights + optimizer state)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Train the model first (Step 6).")
    return keras.models.load_model(path)
