"""
Step 7 - Evaluation and comparison with baselines / Etape 7 - Evaluation et comparaison.

Run:  python -m src.forecast.evaluate --split val            (model selection, as often as needed)
      python -m src.forecast.evaluate --split test --final   (ONCE, at the very end)

All errors are computed in ORIGINAL units (weekly sales), after the inverse transform of the
LSTM output. Baselines see exactly the same information as the LSTM: history up to the origin week.

Models compared
  lstm            : the trained network
  naive           : last observed week, repeated
  moving_average  : mean of the last `ma_window` weeks, repeated
  seasonal_naive  : same week one year earlier (falls back to naive if that week does not exist)

Metrics: MAE, RMSE, MAPE (only where actual > 0, coverage reported), WAPE = sum|error| / sum(actual),
sMAPE, bias. MAE/RMSE in units are dominated by the biggest products, so scale-free summaries
(pooled WAPE, mean of per-product WAPE) and the number of products beaten are reported too.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.forecast import data_loader as dl
from src.forecast import sequence_generator as sg

logger = logging.getLogger(__name__)
EVAL_DEFAULTS = {"ma_window": 4, "season_length": 52}
BASELINES = ("naive", "moving_average", "seasonal_naive")


def get_eval_params(config: dict) -> dict:
    """Return the `evaluation` settings of config.yaml, completed with defaults."""
    return {**EVAL_DEFAULTS, **(config.get("evaluation") or {})}


# --------------------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred) -> dict:
    """Forecast-error metrics for two arrays of the same shape (any dimension).

    MAPE ignores points where the actual value is 0 (division by zero) and `mape_coverage_pct`
    tells which share of the points it uses. WAPE = 100 * sum|e| / sum(actual).
    """
    yt, yp = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    if yt.shape != yp.shape or yt.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and have the same size.")
    err = yp - yt
    pos = yt > 0
    denom = np.abs(yt) + np.abs(yp)
    smape_terms = np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0.0)
    return {
        "n": int(yt.size),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE": float((np.abs(err[pos]) / yt[pos]).mean() * 100) if pos.any() else float("nan"),
        "MAPE_coverage_pct": float(pos.mean() * 100),
        "WAPE": float(np.abs(err).sum() / yt.sum() * 100) if yt.sum() > 0 else float("nan"),
        "sMAPE": float(smape_terms.mean() * 100),
        "bias": float(err.mean()),
    }


# --------------------------------------------------------------------------- baselines
def build_sales_lookup(weekly: pd.DataFrame) -> pd.DataFrame:
    """Pivot the weekly table: index = Monday, columns = product, values = sales (NaN = no data)."""
    pivot = weekly.pivot_table(index="week", columns="product_id", values="sales", aggfunc="sum")
    pivot.index = pd.DatetimeIndex(pivot.index).astype("datetime64[ns]")
    return pivot.reindex(pd.date_range(pivot.index.min(), pivot.index.max(), freq="7D"))


def _positions(lookup: pd.DataFrame, meta: pd.DataFrame, horizon: int):
    """Row/column positions in `lookup` of the origin weeks and of the target weeks (n, horizon)."""
    origin = pd.DatetimeIndex(pd.to_datetime(meta["origin_week"])).astype("datetime64[ns]")
    cols = lookup.columns.get_indexer(meta["product_id"].astype(str))
    if (cols < 0).any():
        raise ValueError("Some products of the samples are missing from the weekly table.")
    rows_origin = lookup.index.get_indexer(origin)
    rows_target = np.stack([lookup.index.get_indexer(origin + pd.Timedelta(days=7 * h)) for h in range(1, horizon + 1)], axis=1)
    return cols, rows_origin, rows_target, origin


def actual_units(lookup: pd.DataFrame, meta: pd.DataFrame, horizon: int) -> np.ndarray:
    """Actual weekly sales of the `horizon` target weeks of every sample, shape (n, horizon)."""
    cols, _, rows_target, _ = _positions(lookup, meta, horizon)
    if (rows_target < 0).any():
        raise ValueError("Target weeks outside the weekly table.")
    values = lookup.to_numpy(float)[rows_target, cols[:, None]]
    if np.isnan(values).any():
        raise ValueError("Missing actual values for some target weeks.")
    return values


def baseline_forecasts(lookup: pd.DataFrame, meta: pd.DataFrame, horizon: int,
                       ma_window: int = 4, season_length: int = 52) -> dict:
    """Naive, moving-average and seasonal-naive forecasts, shape (n, horizon) each.

    Only weeks <= origin_week are read: no baseline uses the future.
    Output also contains "seasonal_fallback_pct": share of seasonal forecasts that fell back to naive.
    """
    cols, rows_origin, rows_target, origin = _positions(lookup, meta, horizon)
    data = lookup.to_numpy(float)
    n = len(meta)
    last = data[rows_origin, cols]
    if np.isnan(last).any():
        raise ValueError("Missing value at the origin week for some samples.")

    ma = np.empty(n)
    for i in range(n):
        window = data[max(0, rows_origin[i] - ma_window + 1): rows_origin[i] + 1, cols[i]]
        ma[i] = np.nanmean(window)

    seasonal = np.empty((n, horizon))
    fallback = 0
    for h in range(1, horizon + 1):
        rows_prev = lookup.index.get_indexer(origin + pd.Timedelta(days=7 * h) - pd.Timedelta(days=7 * season_length))
        valid = rows_prev >= 0
        vals = np.where(valid, data[np.where(valid, rows_prev, 0), cols], np.nan)
        missing = np.isnan(vals)
        seasonal[:, h - 1] = np.where(missing, last, vals)
        fallback += int(missing.sum())

    return {
        "naive": np.tile(last[:, None], (1, horizon)),
        "moving_average": np.tile(ma[:, None], (1, horizon)),
        "seasonal_naive": seasonal,
        "seasonal_fallback_pct": fallback / (n * horizon) * 100,
    }


def predict_lstm(model, seq: dict, scaler, model_meta: Optional[dict] = None) -> np.ndarray:
    """LSTM forecasts in ORIGINAL units, shape (n, horizon).

    model_meta (models/model_meta.json) says how the output must be decoded
    ("delta" outputs are added back to the last observed level).
    """
    from src.forecast import model as md

    meta = model_meta or {"target_mode": "level", "level_idx": 0}
    out = model.predict(md.model_inputs(model, seq), verbose=0)
    scaled = md.decode_prediction(out, seq["X"], meta["target_mode"], meta.get("level_idx", 0))
    return scaler.inverse_sales(seq["meta"]["product_id"].to_numpy(), scaled)


# --------------------------------------------------------------------------- comparison
def evaluate_models(y_true: np.ndarray, preds: Dict[str, np.ndarray], meta: pd.DataFrame) -> dict:
    """Compute all metrics for every model: pooled, per horizon, per product.

    Output keys: overall, by_horizon, by_product (DataFrame), macro_wape (mean of per-product WAPE),
    products_beaten (for each baseline, number of products where the LSTM has a lower MAE).
    """
    horizon = y_true.shape[1]
    overall = {m: regression_metrics(y_true, p) for m, p in preds.items()}
    by_h = {m: [regression_metrics(y_true[:, h], p[:, h]) for h in range(horizon)] for m, p in preds.items()}

    products = meta["product_id"].to_numpy()
    rows = []
    for pid in sorted(set(products)):
        mask = products == pid
        row = {"product_id": pid, "n_samples": int(mask.sum()), "mean_weekly_sales": float(y_true[mask].mean())}
        for m, p in preds.items():
            r = regression_metrics(y_true[mask], p[mask])
            row[f"MAE_{m}"], row[f"WAPE_{m}"] = r["MAE"], r["WAPE"]
        rows.append(row)
    by_product = pd.DataFrame(rows)
    names = list(preds)
    by_product["best_model"] = by_product[[f"MAE_{m}" for m in names]].idxmin(axis=1).str.replace("MAE_", "", regex=False)

    macro = {m: float(by_product[f"WAPE_{m}"].mean()) for m in names}
    beaten = {}
    if "lstm" in preds:
        for b in [n for n in names if n != "lstm"]:
            beaten[b] = int((by_product["MAE_lstm"] < by_product[f"MAE_{b}"]).sum())
    return {"overall": overall, "by_horizon": by_h, "by_product": by_product, "macro_wape": macro,
            "products_beaten": beaten, "n_products": int(len(by_product))}


def summarize(result: dict) -> list:
    """Human-readable verdict lines (which model is best and by how much)."""
    lines = []
    pooled = {m: v["WAPE"] for m, v in result["overall"].items()}
    macro = result["macro_wape"]
    best_pooled, best_macro = min(pooled, key=pooled.get), min(macro, key=macro.get)
    lines.append(f"Best model by pooled WAPE: {best_pooled} ({pooled[best_pooled]:.1f}%)")
    lines.append(f"Best model by mean per-product WAPE: {best_macro} ({macro[best_macro]:.1f}%)")
    if "lstm" in pooled:
        for b in [m for m in pooled if m != "lstm"]:
            gain = (1 - pooled["lstm"] / pooled[b]) * 100
            lines.append(f"LSTM vs {b}: pooled WAPE {pooled['lstm']:.1f}% vs {pooled[b]:.1f}% "
                         f"({gain:+.1f}% relative), beats it on {result['products_beaten'][b]}/{result['n_products']} products (MAE)")
    return lines


# --------------------------------------------------------------------------- plots
def plot_comparison(result: dict, path: Path) -> None:
    """Bar charts: WAPE per model for each forecast horizon, and per-product WAPE averaged."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(result["overall"])
    horizon = len(next(iter(result["by_horizon"].values())))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.8 / len(models)
    for k, m in enumerate(models):
        vals = [result["by_horizon"][m][h]["WAPE"] for h in range(horizon)]
        axes[0].bar(np.arange(horizon) + k * width, vals, width, label=m)
    axes[0].set_xticks(np.arange(horizon) + 0.4 - width / 2)
    axes[0].set_xticklabels([f"week +{h + 1}" for h in range(horizon)])
    axes[0].set_ylabel("WAPE (%)"); axes[0].set_title("Pooled WAPE by forecast horizon"); axes[0].legend()
    axes[1].bar(models, [result["macro_wape"][m] for m in models], color="tab:gray")
    axes[1].set_ylabel("WAPE (%)"); axes[1].set_title("Mean of per-product WAPE")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)


def plot_forecast_grid(preds_table: pd.DataFrame, path: Path, title: str) -> None:
    """One small chart per product: actual vs LSTM vs naive for the 1-week-ahead forecast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    products = sorted(preds_table["product_id"].unique())
    ncols = 4
    nrows = int(np.ceil(len(products) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, pid in zip(axes.ravel(), products):
        g = preds_table[(preds_table["product_id"] == pid) & (preds_table["horizon"] == 1)].sort_values("target_week")
        ax.axis("on")
        ax.plot(g["target_week"], g["actual"], color="black", linewidth=1.2, label="actual")
        ax.plot(g["target_week"], g["lstm"], color="tab:red", linewidth=1, label="LSTM")
        ax.plot(g["target_week"], g["naive"], color="tab:blue", linewidth=0.8, alpha=0.6, label="naive")
        ax.set_title(pid, fontsize=8); ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(title); fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- pipeline
def compute_evaluation(config: dict, split: str, prepared: dict, model, model_meta: dict,
                       weekly: pd.DataFrame) -> dict:
    """Evaluate the LSTM and the baselines on one split (nothing is written to disk)."""
    params = get_eval_params(config)
    seq, info, scaler = prepared["sequences"][split], prepared["info"], prepared["scaler"]
    if len(seq["X"]) == 0:
        raise ValueError(f"No samples in the {split} split.")
    meta, horizon = seq["meta"], info["horizon"]
    lookup = build_sales_lookup(weekly)

    y_true = actual_units(lookup, meta, horizon)
    check = scaler.inverse_sales(meta["product_id"].to_numpy(), seq["y"])
    if not np.allclose(check, y_true, rtol=1e-3, atol=1e-2):
        raise RuntimeError("Inconsistency between the scaled targets and the weekly table (scaler / data mismatch).")

    base = baseline_forecasts(lookup, meta, horizon, params["ma_window"], params["season_length"])
    preds = {"lstm": predict_lstm(model, seq, scaler, model_meta), **{b: base[b] for b in BASELINES}}
    result = evaluate_models(y_true, preds, meta)
    result["seasonal_fallback_pct"] = base["seasonal_fallback_pct"]
    result["verdict"] = summarize(result)

    rows = []
    for i, r in enumerate(meta.itertuples(index=False)):
        for h in range(horizon):
            rows.append({"product_id": r.product_id, "origin_week": r.origin_week, "horizon": h + 1,
                         "target_week": pd.Timestamp(r.origin_week) + pd.Timedelta(days=7 * (h + 1)),
                         "actual": y_true[i, h], **{m: p[i, h] for m, p in preds.items()}})
    result["predictions"] = pd.DataFrame(rows)
    result["n_samples"] = int(len(meta))
    return result


def run_evaluation(config: Optional[dict] = None, split: str = "val", final: bool = False,
                   root: Optional[Path] = None, prepared: Optional[dict] = None, model=None,
                   weekly: Optional[pd.DataFrame] = None, tag: str = "") -> dict:
    """Evaluate the LSTM and the baselines on one split and write the reports.

    The test split can only be evaluated with final=True: use it once, after all tuning is done,
    otherwise the reported numbers are no longer an honest estimate of future performance.
    Writes outputs/evaluation_<split>.json, evaluation_<split>_by_product.csv,
    predictions_<split>.csv and plots/07_*.png
    """
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    if split == "test" and not final:
        raise ValueError("The test set is reserved for the FINAL evaluation. "
                         "Tune on --split val; run --split test --final only once at the end.")
    from src.forecast import model as md
    from src.forecast.train import load_model_meta

    config = config or dl.load_config()
    root = Path(root) if root else dl.PROJECT_ROOT
    prepared = prepared or sg.load_prepared(config, root=root)
    outputs = root / config["paths"]["outputs_dir"]
    models_dir = root / config["paths"]["models_dir"]
    if weekly is None:
        weekly = pd.read_csv(outputs / config.get("preprocessing", {}).get("weekly_file", "sales_weekly.csv"), parse_dates=["week"])
    model_file = Path(md.get_model_params(config)["model_file"])
    if model is None:
        model = md.load_model(models_dir / f"{model_file.stem}{tag}{model_file.suffix}")
    model_meta = load_model_meta(models_dir / f"model_meta{tag}.json")

    result = compute_evaluation(config, split, prepared, model, model_meta, weekly)
    table = result["predictions"]

    params = get_eval_params(config)
    outputs.mkdir(parents=True, exist_ok=True)
    table.to_csv(outputs / f"predictions_{split}.csv", index=False)
    result["by_product"].to_csv(outputs / f"evaluation_{split}_by_product.csv", index=False)
    summary = {k: v for k, v in result.items() if k not in ("by_product", "predictions")}
    summary.update({"split": split, "ma_window": params["ma_window"], "season_length": params["season_length"],
                    "final": bool(final), "target_mode": model_meta["target_mode"]})
    (outputs / f"evaluation_{split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_comparison(result, outputs / "plots" / f"07_model_comparison_{split}.png")
    plot_forecast_grid(table, outputs / "plots" / f"07_forecast_by_product_{split}.png",
                       f"1-week-ahead forecast, {split} period (units per week)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the LSTM against the baselines.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final", action="store_true", help="required for the test split (use it once)")
    args = parser.parse_args()
    try:
        res = run_evaluation(split=args.split, final=args.final)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}")

    print(f"\n=== Evaluation on the {args.split.upper()} split ({res['n_products']} products) ===")
    cols = ["MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "bias"]
    table = pd.DataFrame(res["overall"]).T[cols + ["MAPE_coverage_pct"]].round(2)
    print(table.to_string())
    print("\nPooled WAPE (%) by horizon:")
    print(pd.DataFrame({m: [h["WAPE"] for h in v] for m, v in res["by_horizon"].items()},
                       index=[f"week +{i + 1}" for i in range(len(next(iter(res['by_horizon'].values()))))]).round(1).to_string())
    print(f"\nSeasonal naive fell back to naive for {res['seasonal_fallback_pct']:.1f}% of forecasts")
    print()
    for line in res["verdict"]:
        print(line)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
