"""
Step 6d - Ensemble of LSTMs / Etape 6d - Ensemble de LSTM (VALIDATION only).

Run:  python -m src.forecast.ensemble                    (5 seeds: 42..46)
      python -m src.forecast.ensemble --seeds 1 2 3

Why / Pourquoi: one network trained on ~4,800 overlapping windows changes a little with the random
seed (initial weights, shuffling). Averaging N networks with the same architecture but different
seeds removes a part of this random noise. It is still a pure LSTM model.

What it does:
  1. trains one LSTM per seed (same config.yaml, only the seed changes); files are tagged
     `_ens_s<seed>` so the main model is never overwritten;
  2. forecasts in ORIGINAL units, then averages the members;
  3. compares on the VALIDATION split: ensemble ("lstm"), the member of the first seed
     ("lstm_single"), the 3 baselines and, as a DIAGNOSTIC only, a 50/50 blend of the ensemble with
     the seasonal naive ("diag_blend_seasonal");
  4. reports the spread between members (seed variance).

The test split is NOT used (this module has no --split option).
Writes: models/lstm_demand_model_ens_s<seed>.keras (+ model_meta), models/ensemble_meta.json,
        outputs/ensemble_val.json, ensemble_val_by_product.csv, ensemble_members_val.csv,
        plots/07_ensemble_val.png
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.forecast import data_loader as dl
from src.forecast import evaluate as ev
from src.forecast import sequence_generator as sg
from src.forecast import train as tr

logger = logging.getLogger(__name__)
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
BLEND_NAME = "diag_blend_seasonal"


def get_ensemble_seeds(config: dict, seeds: Optional[Sequence[int]] = None) -> List[int]:
    """Seeds to use: argument > config['ensemble']['seeds'] > DEFAULT_SEEDS. Needs >= 2 distinct seeds."""
    chosen = list(seeds) if seeds else list((config.get("ensemble") or {}).get("seeds", DEFAULT_SEEDS))
    chosen = [int(s) for s in chosen]
    if len(chosen) < 2 or len(set(chosen)) != len(chosen):
        raise ValueError("An ensemble needs at least 2 different seeds.")
    return chosen


def average_forecasts(forecasts: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of several forecast arrays of identical shape (n, horizon), in original units."""
    arrs = [np.asarray(f, float) for f in forecasts]
    if not arrs or any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError("All member forecasts must have the same shape.")
    return np.mean(arrs, axis=0)


def blend(a: np.ndarray, b: np.ndarray, weight_a: float = 0.5) -> np.ndarray:
    """weight_a * a + (1 - weight_a) * b."""
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight_a must be between 0 and 1.")
    return weight_a * np.asarray(a, float) + (1 - weight_a) * np.asarray(b, float)


def train_ensemble(config: Optional[dict] = None, seeds: Optional[Sequence[int]] = None,
                   prepared: Optional[dict] = None, root: Optional[Path] = None, verbose: int = 0) -> List[dict]:
    """Train one LSTM per seed. Returns a list of dicts: seed, tag, model, model_meta, summary."""
    config = config or dl.load_config()
    root = Path(root) if root else dl.PROJECT_ROOT
    seeds = get_ensemble_seeds(config, seeds)
    prepared = prepared or sg.load_prepared(config, root=root)
    members = []
    for k, seed in enumerate(seeds, 1):
        start = time.time()
        tag = f"_ens_s{seed}"
        logger.info("[%d/%d] training member with seed %d ...", k, len(seeds), seed)
        trained = tr.train_model({**config, "seed": seed}, prepared=prepared, root=root, verbose=verbose, tag=tag)
        logger.info("    done in %.1f min (best epoch %d, val_loss %.4f)", (time.time() - start) / 60,
                    trained["summary"]["best_epoch"], trained["summary"]["best_val_loss"])
        members.append({"seed": seed, "tag": tag, "model": trained["model"],
                        "model_meta": trained["model_meta"], "summary": trained["summary"]})
    return members


def predict_members(members: Sequence[dict], seq: dict, scaler) -> List[np.ndarray]:
    """Forecast (original units) of every member for the samples `seq`."""
    return [ev.predict_lstm(m["model"], seq, scaler, m["model_meta"]) for m in members]


def evaluate_ensemble(config: dict, prepared: dict, members: Sequence[dict], weekly: pd.DataFrame,
                      split: str = "val") -> dict:
    """Compare ensemble, single member, baselines and the diagnostic blend. Only 'val' is allowed here."""
    if split != "val":
        raise ValueError("The ensemble is evaluated on the validation split only; "
                         "the test split is reserved for the final evaluation of the chosen model.")
    params = ev.get_eval_params(config)
    seq, info, scaler = prepared["sequences"]["val"], prepared["info"], prepared["scaler"]
    if len(seq["X"]) == 0:
        raise ValueError("No samples in the validation split.")
    meta, horizon = seq["meta"], info["horizon"]
    lookup = ev.build_sales_lookup(weekly)
    y_true = ev.actual_units(lookup, meta, horizon)
    base = ev.baseline_forecasts(lookup, meta, horizon, params["ma_window"], params["season_length"])

    member_preds = predict_members(members, seq, scaler)
    ensemble = average_forecasts(member_preds)
    preds = {"lstm": ensemble, "lstm_single": member_preds[0],
             **{b: base[b] for b in ev.BASELINES}, BLEND_NAME: blend(ensemble, base["seasonal_naive"], 0.5)}
    result = ev.evaluate_models(y_true, preds, meta)
    result["verdict"] = ev.summarize(result)

    rows = []
    for m, p in zip(members, member_preds):
        r = ev.regression_metrics(y_true, p)
        rows.append({"seed": m["seed"], "best_epoch": m["summary"]["best_epoch"],
                     "val_loss": m["summary"]["best_val_loss"], "WAPE": r["WAPE"], "MAE": r["MAE"],
                     "RMSE": r["RMSE"], "bias": r["bias"]})
    result["members"] = pd.DataFrame(rows)
    result["n_samples"] = int(len(meta))
    return result


def run_ensemble(config: Optional[dict] = None, seeds: Optional[Sequence[int]] = None,
                 root: Optional[Path] = None, prepared: Optional[dict] = None,
                 weekly: Optional[pd.DataFrame] = None, verbose: int = 0) -> dict:
    """Train the members, evaluate on validation and write the reports (see module docstring)."""
    config = config or dl.load_config()
    root = Path(root) if root else dl.PROJECT_ROOT
    prepared = prepared or sg.load_prepared(config, root=root)
    outputs = root / config["paths"]["outputs_dir"]
    models_dir = root / config["paths"]["models_dir"]
    if weekly is None:
        weekly = pd.read_csv(outputs / config.get("preprocessing", {}).get("weekly_file", "sales_weekly.csv"),
                             parse_dates=["week"])

    members = train_ensemble(config, seeds, prepared, root, verbose)
    result = evaluate_ensemble(config, prepared, members, weekly)

    outputs.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    stem, suffix = Path(config["model"].get("model_file", "lstm_demand_model.keras")).stem, \
        Path(config["model"].get("model_file", "lstm_demand_model.keras")).suffix
    ens_meta = {"seeds": [m["seed"] for m in members],
                "files": [f"{stem}{m['tag']}{suffix}" for m in members],
                "meta_files": [f"model_meta{m['tag']}.json" for m in members],
                "combine": "mean of the member forecasts in original units"}
    (models_dir / "ensemble_meta.json").write_text(json.dumps(ens_meta, indent=2), encoding="utf-8")

    result["members"].to_csv(outputs / "ensemble_members_val.csv", index=False)
    result["by_product"].to_csv(outputs / "ensemble_val_by_product.csv", index=False)
    summary = {k: v for k, v in result.items() if k not in ("by_product", "members")}
    summary["members"] = result["members"].to_dict(orient="records")
    summary["split"] = "val"
    (outputs / "ensemble_val.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ev.plot_comparison(result, outputs / "plots" / "07_ensemble_val.png")
    result["ensemble_meta"] = ens_meta
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a seed ensemble of LSTMs and compare it on validation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    try:
        res = run_ensemble(seeds=args.seeds)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}")

    print(f"\n=== Members (validation) ===")
    print(res["members"].round(4).to_string(index=False))
    w = res["members"]["WAPE"]
    print(f"\nMember pooled WAPE: min {w.min():.2f}% / mean {w.mean():.2f}% / max {w.max():.2f}%  (spread = seed noise)")

    print(f"\n=== Ensemble vs baselines, VALIDATION ({res['n_products']} products) ===")
    cols = ["MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "bias"]
    print(pd.DataFrame(res["overall"]).T[cols].round(2).to_string())
    print("\nPooled WAPE (%) by horizon:")
    n_h = len(next(iter(res["by_horizon"].values())))
    print(pd.DataFrame({m: [h["WAPE"] for h in v] for m, v in res["by_horizon"].items()},
                       index=[f"week +{i + 1}" for i in range(n_h)]).round(1).to_string())
    print()
    for line in res["verdict"]:
        print(line)
    print(f"\n('{BLEND_NAME}' is a diagnostic only. The test set was NOT used.)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
