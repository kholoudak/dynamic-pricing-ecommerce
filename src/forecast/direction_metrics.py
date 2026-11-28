"""
Classification-style metrics on the DIRECTION of demand / Metriques de classification sur la direction.

Run:  python -m src.forecast.direction_metrics

Why: accuracy, precision, recall, F-score, specificity and type I / type II errors exist only for a
yes/no question. The LSTM predicts a quantity (regression), so we ask a yes/no question built from it:

    "Will demand in the target week be HIGHER than the last observed week?"

  actual class    = actual sales(target week)   > sales(origin week)
  predicted class = forecast(target week)       > sales(origin week)
  positive class  = "increase"  (so: type I error = false alarm of an increase,
                                     type II error = a real increase that was missed)

Reads outputs/predictions_test.csv (written once by final_evaluation) and outputs/sales_weekly.csv.
Nothing is retrained and the test set is not re-evaluated. Summaries only: do not use them to select the model.
Note: the naive forecast equals the last observed week, so it never predicts an increase (precision is undefined).
Writes outputs/direction_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.forecast import data_loader as dl


NON_MODEL_COLUMNS = {"product_id", "origin_week", "horizon", "target_week", "actual"}


def model_columns(df: pd.DataFrame) -> list:
    """Columns of the prediction table that hold forecasts (everything except keys and the actual value)."""
    return [c for c in df.columns if c not in NON_MODEL_COLUMNS]


def _div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def confusion_metrics(y_true, y_pred) -> dict:
    """Binary metrics, positive class = True. Rates are fractions between 0 and 1 (nan if undefined)."""
    t, p = np.asarray(y_true, bool).ravel(), np.asarray(y_pred, bool).ravel()
    if t.shape != p.shape or t.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and have the same size.")
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    tn, fn = int((~t & ~p).sum()), int((t & ~p).sum())
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = _div(2 * precision * recall, precision + recall) if np.isfinite(precision) and np.isfinite(recall) else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": int(t.size),
            "accuracy": _div(tp + tn, t.size), "precision": precision, "recall": recall, "f1": f1,
            "specificity": _div(tn, tn + fp),
            "type1_error": _div(fp, fp + tn),    # false positive rate = 1 - specificity
            "type2_error": _div(fn, fn + tp)}    # false negative rate = 1 - recall


def add_last_observed(pred: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """Add `last_obs` = sales of the origin week (the last week the model saw) to a predictions table."""
    w = weekly[["product_id", "week", "sales"]].rename(columns={"week": "origin_week", "sales": "last_obs"})
    w = w.assign(product_id=w["product_id"].astype(str))
    p = pred.assign(product_id=pred["product_id"].astype(str), origin_week=pd.to_datetime(pred["origin_week"]))
    out = p.merge(w.assign(origin_week=pd.to_datetime(w["origin_week"])), on=["product_id", "origin_week"], how="left")
    if out["last_obs"].isna().any():
        raise ValueError("Some origin weeks are missing from the weekly table.")
    return out


def compute_direction(table: pd.DataFrame) -> Dict[str, dict]:
    """Per model: confusion-matrix metrics of the 'increase vs last observed week' question."""
    if not {"actual", "last_obs"}.issubset(table.columns):
        raise ValueError("The table needs the columns 'actual' and 'last_obs'.")
    truth = table["actual"] > table["last_obs"]
    out = {}
    for m in model_columns(table.drop(columns=["last_obs"])):
        res = confusion_metrics(truth, table[m] > table["last_obs"])
        res["accuracy_by_horizon"] = {int(h): confusion_metrics(truth[g.index], g[m] > g["last_obs"])["accuracy"]
                                      for h, g in table.groupby("horizon")}
        out[m] = res
    out["_share_of_real_increases"] = float(truth.mean())
    return out


def main(root: Optional[Path] = None) -> None:
    config = dl.load_config()
    root = Path(root) if root else dl.PROJECT_ROOT
    outputs = root / config["paths"]["outputs_dir"]
    pred_path = outputs / "predictions_test.csv"
    weekly_path = outputs / config.get("preprocessing", {}).get("weekly_file", "sales_weekly.csv")
    for p in (pred_path, weekly_path):
        if not p.exists():
            raise SystemExit(f"{p} not found: run the previous steps first.")
    table = add_last_observed(pd.read_csv(pred_path, dtype={"product_id": str}),
                              pd.read_csv(weekly_path, dtype={"product_id": str}))
    res = compute_direction(table)
    share = res.pop("_share_of_real_increases")
    cols = ["accuracy", "precision", "recall", "f1", "specificity", "type1_error", "type2_error"]
    df = pd.DataFrame({m: {c: v[c] for c in cols} for m, v in res.items()}).T
    counts = pd.DataFrame({m: {c: v[c] for c in ("tp", "fp", "tn", "fn")} for m, v in res.items()}).T
    print("=== FINAL TEST: 'will demand increase vs the last observed week?' (positive class = increase) ===")
    print(f"Share of real increases in the test set: {share * 100:.1f}%  (baseline accuracy of always guessing the majority class: "
          f"{max(share, 1 - share) * 100:.1f}%)\n")
    print((df * 100).round(1).to_string())
    print("\n(values in %; type I = false alarm of an increase, type II = missed increase)\n")
    print(counts.to_string())
    res["_share_of_real_increases"] = share
    (outputs / "direction_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nSaved: outputs/direction_metrics.json  (summaries only; do not use them to select the model)")


if __name__ == "__main__":
    main()
