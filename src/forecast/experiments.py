"""
Step 6c - Controlled experiments on the VALIDATION set / Etape 6c - Experiences sur la validation.

Run:  python -m src.forecast.experiments
      python -m src.forecast.experiments --variants delta delta_seasonal --include-level

Why: the first LSTM (target_mode "level") was worse than the naive baselines on validation.
Each variant below changes ONE idea; all of them use the same data, the same seed, the same split
and are compared on the validation period only. The test set is never read here.

  level               current setting (reference, only with --include-level)
  delta               the network predicts the CHANGE vs the last week (0 = naive forecast)
  delta_seasonal      delta + an input telling what happened one year ago over the same weeks
  delta_seasonal_reg  same + smaller network and stronger dropout (against overfitting)

Writes outputs/experiments_val.csv; each variant is saved under its own file names
(models/lstm_demand_model_exp_<name>.keras, ...), the main model is not touched.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.forecast import data_loader as dl
from src.forecast import evaluate as ev
from src.forecast import sequence_generator as sg
from src.forecast import train as tr

logger = logging.getLogger(__name__)

VARIANTS: Dict[str, dict] = {
    "level": {"forecast": {"target_mode": "level"}, "model": {"use_seasonal_input": False}},
    "delta": {"forecast": {"target_mode": "delta"}, "model": {"use_seasonal_input": False}},
    "delta_seasonal": {"forecast": {"target_mode": "delta"}, "model": {"use_seasonal_input": True}},
    "delta_seasonal_reg": {"forecast": {"target_mode": "delta"},
                           "model": {"use_seasonal_input": True, "lstm_units": [32, 16], "dropout": 0.3}},
}
DEFAULT_ORDER = ["delta", "delta_seasonal", "delta_seasonal_reg"]


def apply_variant(config: dict, name: str) -> dict:
    """Return a copy of the config with the overrides of variant `name`."""
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}. Choose among {list(VARIANTS)}.")
    ov = VARIANTS[name]
    return {**config, "forecast": {**config["forecast"], **ov["forecast"]}, "model": {**config["model"], **ov["model"]}}


def summary_row(label: str, res: dict, extra: Optional[dict] = None) -> dict:
    """One line of the comparison table for the model `label` ("lstm" or a baseline name)."""
    o = res["overall"][label]
    row = {"model": label, "WAPE_pooled": o["WAPE"], "WAPE_mean_per_product": res["macro_wape"][label],
           "MAE": o["MAE"], "RMSE": o["RMSE"], "MAPE": o["MAPE"], "bias": o["bias"]}
    row.update(extra or {})
    return row


def run_experiments(config: Optional[dict] = None, root: Optional[Path] = None, variants: Optional[List[str]] = None,
                    include_level: bool = False, prepared: Optional[dict] = None,
                    weekly: Optional[pd.DataFrame] = None, verbose: int = 0) -> pd.DataFrame:
    """Train every variant and compare them (and the baselines) on the validation split."""
    config = config or dl.load_config()
    root = Path(root) if root else dl.PROJECT_ROOT
    names = list(variants or DEFAULT_ORDER)
    if include_level and "level" not in names:
        names.insert(0, "level")
    prepared = prepared or sg.load_prepared(config, root=root)
    outputs = root / config["paths"]["outputs_dir"]
    if weekly is None:
        weekly = pd.read_csv(outputs / config.get("preprocessing", {}).get("weekly_file", "sales_weekly.csv"), parse_dates=["week"])

    rows, reference = [], None
    for k, name in enumerate(names, 1):
        start = time.time()
        logger.info("[%d/%d] training variant '%s' ...", k, len(names), name)
        cfg = apply_variant(config, name)
        trained = tr.train_model(cfg, prepared=prepared, root=root, verbose=verbose, tag=f"_exp_{name}")
        res = ev.compute_evaluation(cfg, "val", prepared, trained["model"], trained["model_meta"], weekly)
        s = trained["summary"]
        rows.append(summary_row("lstm", res, {
            "variant": name, "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"],
            "beats_naive": f"{res['products_beaten']['naive']}/{res['n_products']}",
            "beats_seasonal": f"{res['products_beaten']['seasonal_naive']}/{res['n_products']}",
            "minutes": round((time.time() - start) / 60, 1)}))
        reference = reference or res
        logger.info("    done in %.1f min: pooled WAPE %.1f%%", (time.time() - start) / 60, res["overall"]["lstm"]["WAPE"])

    for b in ev.BASELINES:
        rows.append(summary_row(b, reference, {"variant": f"baseline:{b}"}))
    table = pd.DataFrame(rows).set_index("variant")
    table = table.sort_values("WAPE_pooled")
    outputs.mkdir(parents=True, exist_ok=True)
    table.to_csv(outputs / "experiments_val.csv")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LSTM variants on the validation split.")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)
    parser.add_argument("--include-level", action="store_true", help="also retrain the current 'level' model as reference")
    args = parser.parse_args()
    try:
        table = run_experiments(variants=args.variants, include_level=args.include_level)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}")
    cols = ["model", "WAPE_pooled", "WAPE_mean_per_product", "MAE", "RMSE", "MAPE", "bias"]
    extra = [c for c in ("best_epoch", "beats_naive", "beats_seasonal", "minutes") if c in table.columns]
    print("\n=== Validation results (sorted by pooled WAPE, lower is better) ===")
    print(table[cols + extra].round(2).to_string())
    print("\nThe test set was NOT used. Saved: outputs/experiments_val.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
