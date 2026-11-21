"""
Step 6b - Training / Etape 6b - Entrainement.

Run:  python -m src.forecast.train

- trains on the TRAIN samples, monitors the VALIDATION loss (the test set is never used here);
- EarlyStopping (restores the best weights), ModelCheckpoint (best model saved to
  models/lstm_demand_model.keras), ReduceLROnPlateau, CSV history;
- target_mode (config forecast.target_mode):
    "level": the network predicts the scaled log-sales of the next weeks;
    "delta": it predicts their CHANGE relative to the last observed week, so an output of 0 is
             exactly the naive forecast and the network only has to learn corrections;
- samples are shuffled between epochs: each window already contains its own past, so shuffling
  the samples does not mix time inside a window;
- models/model_meta<tag>.json records how the model was trained (target mode, inputs, features),
  so that evaluation and prediction decode its output correctly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.forecast import data_loader as dl
from src.forecast import model as md
from src.forecast import sequence_generator as sg

logger = logging.getLogger(__name__)


def plot_history(history: pd.DataFrame, path: Path) -> None:
    """Save the training / validation loss curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (col, title) in zip(axes, [("loss", "Loss"), ("mae", "MAE (scaled log-sales)")]):
        ax.plot(history.index + 1, history[col], label="train")
        ax.plot(history.index + 1, history[f"val_{col}"], label="validation")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_model_meta(path: Path) -> dict:
    """Read models/model_meta.json. Models trained before it existed are 'level', no seasonal input."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"target_mode": "level", "level_idx": 0, "use_seasonal_input": False}


def train_model(config: Optional[dict] = None, prepared: Optional[dict] = None,
                root: Optional[Path] = None, verbose: int = 2, tag: str = "") -> dict:
    """Build, train and save the LSTM.

    Input : config (default config.yaml); prepared = output of sg.load_prepared (default: loaded
            from disk); root = project root (default: the real one); tag = suffix added to every
            output file name (used by the experiments so that they do not overwrite the main model).
    Output: dict with the model, the Keras history, model_meta and a summary.
    Writes: models/<model_file><tag>, models/model_meta<tag>.json, outputs/training_history<tag>.csv,
            outputs/training_summary<tag>.json, outputs/plots/06_training_loss<tag>.png
    """
    import keras

    config = config or dl.load_config()
    params = md.get_model_params(config)
    mode = md.get_target_mode(config)
    root = Path(root) if root else dl.PROJECT_ROOT
    prepared = prepared or sg.load_prepared(config, root=root)
    md.set_seed(config.get("seed", 42))

    seq, info = prepared["sequences"], prepared["info"]
    train, val = seq["train"], seq["val"]
    if len(train["X"]) == 0 or len(val["X"]) == 0:
        raise ValueError("Empty train or validation set: check the split settings and Step 5.")

    level_idx = info["feature_cols"].index("log_sales")
    n_products, window, n_feat = len(info["products"]), train["X"].shape[1], train["X"].shape[2]
    model = md.build_lstm_model(window, n_feat, info["horizon"], n_products, params)
    use_emb, use_seas = md.uses_embedding(model), md.uses_seasonal(model)
    model.summary(print_fn=logger.info)

    models_dir = root / config["paths"]["models_dir"]
    outputs_dir = root / config["paths"]["outputs_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stem, suffix = Path(params["model_file"]).stem, Path(params["model_file"]).suffix
    model_path = models_dir / f"{stem}{tag}{suffix}"

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(params["early_stopping_patience"]),
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=float(params["reduce_lr_factor"]),
                                          patience=int(params["reduce_lr_patience"]), min_lr=1e-5),
    ]
    history = model.fit(
        md.model_inputs(model, train), md.encode_target(train["y"], train["X"], mode, level_idx),
        validation_data=(md.model_inputs(model, val), md.encode_target(val["y"], val["X"], mode, level_idx)),
        epochs=int(params["epochs"]), batch_size=int(params["batch_size"]),
        shuffle=True, callbacks=callbacks, verbose=verbose,
    )

    model_meta = {"target_mode": mode, "level_idx": level_idx, "use_product_embedding": bool(use_emb),
                  "use_seasonal_input": bool(use_seas), "window": int(window), "horizon": int(info["horizon"]),
                  "feature_cols": info["feature_cols"]}
    (models_dir / f"model_meta{tag}.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    hist = pd.DataFrame(history.history)
    hist.to_csv(outputs_dir / f"training_history{tag}.csv", index_label="epoch")
    plot_history(hist, outputs_dir / "plots" / f"06_training_loss{tag}.png")

    best = int(hist["val_loss"].idxmin())
    summary = {
        "epochs_run": int(len(hist)), "best_epoch": best + 1,
        "best_val_loss": float(hist["val_loss"].min()), "train_loss_at_best": float(hist["loss"].iloc[best]),
        "best_val_mae_scaled": float(hist["val_mae"].iloc[best]),
        "n_train_samples": int(len(train["X"])), "n_val_samples": int(len(val["X"])),
        "n_parameters": int(model.count_params()), "uses_product_embedding": bool(use_emb),
        "uses_seasonal_input": bool(use_seas), "target_mode": mode,
        "model_file": str(model_path), "seed": int(config.get("seed", 42)),
    }
    (outputs_dir / f"training_summary{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"model": model, "history": hist, "summary": summary, "model_meta": model_meta}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = train_model()
    for key, value in result["summary"].items():
        print(f"{key}: {value}")
