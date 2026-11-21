"""
Step 9b - FINAL evaluation on the TEST split / Etape 9b - Evaluation finale sur le test (une seule fois).

Run ONCE, when nothing will be tuned any more:
    python -m src.forecast.final_evaluation --final

Evaluates on the last `test_weeks` weeks, never used for training, early stopping or any choice:
  hybrid          : the delivered model = (1 - w) * LSTM ensemble + w * seasonal naive   (w = prediction.seasonal_weight)
  lstm            : the 5-seed LSTM ensemble alone
  lstm_single     : the first member alone (seed variance reference)
  naive / moving_average / seasonal_naive : the baselines

Safety: refuses to run without --final, and refuses a second run (the first result is the honest one;
re-running after looking at it and changing something turns the test set into a second validation set).
Writes outputs/evaluation_test_final.json, evaluation_test_final_by_product.csv, predictions_test.csv,
plots/09_final_test_comparison.png, 09_final_test_forecast_by_product.png and outputs/metrics_summary.json.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.forecast import data_loader as dl
from src.forecast import ensemble as en
from src.forecast import evaluate as ev
from src.forecast import predict as pr
from src.forecast import sequence_generator as sg
from src.forecast import train as tr

logger = logging.getLogger(__name__)
RESULT_FILE = "evaluation_test_final.json"


def compute_final(config: dict, prepared: dict, members: list, weekly: pd.DataFrame, seasonal_weight: float) -> dict:
    """Metrics of hybrid / ensemble / single member / baselines on the TEST split (nothing is written)."""
    params = ev.get_eval_params(config)
    seq, info, scaler = prepared["sequences"]["test"], prepared["info"], prepared["scaler"]
    if len(seq["X"]) == 0:
        raise ValueError("No samples in the test split.")
    meta, horizon = seq["meta"], info["horizon"]
    lookup = ev.build_sales_lookup(weekly)
    y_true = ev.actual_units(lookup, meta, horizon)
    check = scaler.inverse_sales(meta["product_id"].to_numpy(), seq["y"])
    if not np.allclose(check, y_true, rtol=1e-3, atol=1e-2):
        raise RuntimeError("Inconsistency between the scaled targets and the weekly table.")
    base = ev.baseline_forecasts(lookup, meta, horizon, params["ma_window"], params["season_length"])

    member_preds = en.predict_members(members, seq, scaler)
    lstm = en.average_forecasts(member_preds)
    hybrid = np.clip(en.blend(lstm, base["seasonal_naive"], 1.0 - seasonal_weight), 0, None)
    preds = {"hybrid": hybrid, "lstm": lstm, "lstm_single": member_preds[0],
             **{b: base[b] for b in ev.BASELINES}}
    result = ev.evaluate_models(y_true, preds, meta)
    # evaluate_models compares the model called "lstm"; also count the products where the HYBRID wins
    bp = result["by_product"]
    result["hybrid_products_beaten"] = {b: int((bp["MAE_hybrid"] < bp[f"MAE_{b}"]).sum())
                                        for b in ("lstm", "naive", "moving_average", "seasonal_naive")}
    pooled = {m: v["WAPE"] for m, v in result["overall"].items()}
    result["verdict"] = [f"Best model by pooled WAPE: {min(pooled, key=pooled.get)} ({min(pooled.values()):.1f}%)"] + [
        f"hybrid vs {b}: pooled WAPE {pooled['hybrid']:.1f}% vs {pooled[b]:.1f}% "
        f"({(1 - pooled['hybrid'] / pooled[b]) * 100:+.1f}% relative), beats it on "
        f"{result['hybrid_products_beaten'][b]}/{result['n_products']} products (MAE)"
        for b in ("lstm", "naive", "moving_average", "seasonal_naive")]
    rows = []
    for i, r in enumerate(meta.itertuples(index=False)):
        for h in range(horizon):
            rows.append({"product_id": r.product_id, "origin_week": r.origin_week, "horizon": h + 1,
                         "target_week": pd.Timestamp(r.origin_week) + pd.Timedelta(days=7 * (h + 1)),
                         "actual": y_true[i, h], **{m: p[i, h] for m, p in preds.items()}})
    result["predictions"] = pd.DataFrame(rows)
    result["n_samples"] = int(len(meta))
    result["member_wape"] = [float(ev.regression_metrics(y_true, p)["WAPE"]) for p in member_preds]
    return result


def run_final_evaluation(config: Optional[dict] = None, final: bool = False, root: Optional[Path] = None,
                         prepared: Optional[dict] = None, weekly: Optional[pd.DataFrame] = None,
                         allow_rerun: bool = False) -> dict:
    """Run the one-shot test evaluation and write the reports. Requires final=True."""
    if not final:
        raise ValueError("The test set is reserved for the FINAL evaluation: pass --final (use it once).")
    from src.forecast import model as md

    config = config or dl.load_config()
    root = Path(root) if root else dl.PROJECT_ROOT
    outputs = root / config["paths"]["outputs_dir"]
    models_dir = root / config["paths"]["models_dir"]
    if (outputs / RESULT_FILE).exists() and not allow_rerun:
        raise ValueError(f"{outputs / RESULT_FILE} already exists: the test set was already used. "
                         "Re-running after looking at the result makes it a second validation set. "
                         "(--allow-rerun overrides this; say so in the report.)")
    params = pr.get_prediction_params(config)
    prepared = prepared or sg.load_prepared(config, root=root)
    if weekly is None:
        weekly = pd.read_csv(outputs / config.get("preprocessing", {}).get("weekly_file", "sales_weekly.csv"), parse_dates=["week"])
    meta_path = models_dir / "ensemble_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found: run `python -m src.forecast.ensemble` first.")
    em = json.loads(meta_path.read_text(encoding="utf-8"))
    members = [{"model": md.load_model(models_dir / f), "model_meta": tr.load_model_meta(models_dir / m)}
               for f, m in zip(em["files"], em["meta_files"])]

    result = compute_final(config, prepared, members, weekly, params["seasonal_weight"])
    table = result["predictions"]
    outputs.mkdir(parents=True, exist_ok=True)
    table.to_csv(outputs / "predictions_test.csv", index=False)
    result["by_product"].to_csv(outputs / "evaluation_test_final_by_product.csv", index=False)
    summary = {k: v for k, v in result.items() if k not in ("by_product", "predictions")}
    summary.update({"split": "test", "seasonal_weight": params["seasonal_weight"], "seeds": em["seeds"],
                    "rerun_allowed": bool(allow_rerun)})
    (outputs / RESULT_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outputs / "metrics_summary.json").write_text(json.dumps(
        {"test": {m: {k: round(v, 4) for k, v in o.items()} for m, o in result["overall"].items()},
         "delivered_model": "hybrid", "seasonal_weight": params["seasonal_weight"], "seeds": em["seeds"]}, indent=2),
        encoding="utf-8")
    ev.plot_comparison(result, outputs / "plots" / "09_final_test_comparison.png")
    grid = table.assign(lstm=table["hybrid"])  # the grid draws the "lstm" column: show the delivered model
    ev.plot_forecast_grid(grid, outputs / "plots" / "09_final_test_forecast_by_product.png",
                          "Delivered model (hybrid), 1-week-ahead forecast, TEST period (units per week)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot final evaluation on the test split.")
    parser.add_argument("--final", action="store_true", help="required: confirms this is the single test evaluation")
    parser.add_argument("--allow-rerun", action="store_true")
    args = parser.parse_args()
    try:
        res = run_final_evaluation(final=args.final, allow_rerun=args.allow_rerun)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(f"\n=== FINAL evaluation on the TEST split ({res['n_products']} products, {res['n_samples']} samples) ===")
    cols = ["MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "bias"]
    print(pd.DataFrame(res["overall"]).T[cols].round(2).to_string())
    n_h = len(next(iter(res["by_horizon"].values())))
    print("\nPooled WAPE (%) by horizon:")
    print(pd.DataFrame({m: [h["WAPE"] for h in v] for m, v in res["by_horizon"].items()},
                       index=[f"week +{i + 1}" for i in range(n_h)]).round(1).to_string())
    print("\nMean per-product WAPE (%):", {m: round(v, 1) for m, v in res["macro_wape"].items()})
    print("Single-member pooled WAPE (%):", [round(w, 2) for w in res["member_wape"]])
    print()
    for line in res["verdict"]:
        print(line)
    print("\nSaved: outputs/evaluation_test_final*.{json,csv}, predictions_test.csv, metrics_summary.json, plots/09_*.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
