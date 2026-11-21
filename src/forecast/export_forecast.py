"""
Step 9 - Integration file for Agent 3 / Etape 9 - Fichier d'integration pour l'Agent 3.

Run:  python -m src.forecast.export_forecast            (writes + validates outputs/demand_forecast.csv)
      python -m src.forecast.export_forecast --check    (only validates the existing file)

Contract (mandatory columns, one row per product and forecast week):
    product_id   str    same ids as outputs/sales_weekly.csv (product family)
    week         date   Monday (YYYY-MM-DD) of the forecast week
    demand_pred  float  forecast demand, UNITS per week, >= 0
Extra column: horizon (1..4 = weeks after the last week of the data).
Details (LSTM part, seasonal-naive part) are in outputs/demand_forecast_details.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.forecast import data_loader as dl

logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = ["product_id", "week", "demand_pred"]
OUTPUT_COLUMNS = REQUIRED_COLUMNS + ["horizon"]
FORECAST_FILE = "demand_forecast.csv"


def validate_forecast(df: pd.DataFrame, expected_products: Optional[List[str]] = None, horizon: int = 4) -> List[str]:
    """Check the integration contract. Returns the list of problems (empty list = valid)."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing mandatory columns: {missing}"]
    if df.empty:
        return ["the file has no rows"]
    if df[REQUIRED_COLUMNS].isna().any().any():
        problems.append("missing values in mandatory columns")
    pred = pd.to_numeric(df["demand_pred"], errors="coerce")
    if pred.isna().any() or not np.isfinite(pred.dropna()).all():
        problems.append("demand_pred must be numeric and finite")
    elif (pred < 0).any():
        problems.append("demand_pred contains negative values")
    weeks = pd.to_datetime(df["week"], errors="coerce")
    if weeks.isna().any():
        problems.append("week must be a valid date")
    else:
        if (weeks.dt.day_of_week != 0).any():
            problems.append("week must be a Monday")
    if df.duplicated(["product_id", "week"]).any():
        problems.append("duplicated (product_id, week) pairs")
    per_product = df.groupby("product_id")["week"].nunique()
    if (per_product != horizon).any():
        problems.append(f"every product must have exactly {horizon} forecast weeks")
    if not weeks.isna().any() and df.groupby("product_id")["week"].nunique().eq(horizon).all():
        spans = pd.DataFrame({"p": df["product_id"], "w": weeks}).groupby("p")["w"].agg(["min", "max"])
        if not ((spans["max"] - spans["min"]) == pd.Timedelta(days=7 * (horizon - 1))).all():
            problems.append("forecast weeks of a product are not consecutive")
        if spans["min"].nunique() != 1:
            problems.append("products do not start on the same forecast week")
    if expected_products is not None:
        absent = sorted(set(expected_products) - set(df["product_id"].astype(str)))
        extra = sorted(set(df["product_id"].astype(str)) - set(expected_products))
        if absent:
            problems.append(f"products missing from the file: {absent[:5]}")
        if extra:
            problems.append(f"unknown products in the file: {extra[:5]}")
    return problems


def export_forecast(config: Optional[dict] = None, root: Optional[Path] = None, forecaster=None) -> dict:
    """Compute the forecasts and write outputs/demand_forecast.csv (+ details, + plot). Validates before writing.

    Raises ValueError if the contract is not respected (nothing is written in that case).
    """
    from src.forecast.predict import DemandForecaster

    config = config or dl.load_config()
    root = Path(root) if root else dl.PROJECT_ROOT
    f = forecaster or DemandForecaster(config, root)
    table = f.forecast_table()
    out = table[["product_id", "week", "demand_pred", "horizon"]].copy()
    out["week"] = pd.to_datetime(out["week"]).dt.strftime("%Y-%m-%d")
    out["demand_pred"] = out["demand_pred"].round(2)
    problems = validate_forecast(out, f.products, f.horizon)
    if problems:
        raise ValueError("Integration contract violated: " + "; ".join(problems))

    outputs = root / config["paths"]["outputs_dir"]
    outputs.mkdir(parents=True, exist_ok=True)
    out.to_csv(outputs / FORECAST_FILE, index=False)
    table.to_csv(outputs / "demand_forecast_details.csv", index=False)
    plot_forecasts(f.weekly, table, outputs / "plots" / "08_forecast_next_weeks.png")
    info = {"origin_week": str(f.origin_week.date()), "n_products": int(out["product_id"].nunique()),
            "n_rows": int(len(out)), "weeks": sorted(out["week"].unique().tolist()),
            "seasonal_weight": f.params["seasonal_weight"], "n_models": len(f.members),
            "total_demand_by_week": out.groupby("week")["demand_pred"].sum().round(0).to_dict()}
    (outputs / "demand_forecast_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {"forecast": out, "details": table, "info": info}


def plot_forecasts(weekly: pd.DataFrame, table: pd.DataFrame, path: Path, history_weeks: int = 26) -> None:
    """One small chart per product: last weeks of real sales + the 4 forecast weeks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    products = sorted(table["product_id"].unique())
    ncols = 4
    nrows = int(np.ceil(len(products) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, pid in zip(axes.ravel(), products):
        hist = weekly[weekly["product_id"] == pid].sort_values("week").tail(history_weeks)
        fc = table[table["product_id"] == pid].sort_values("week")
        ax.axis("on")
        ax.plot(hist["week"], hist["sales"], color="black", linewidth=1, label="history")
        ax.plot(pd.concat([hist["week"].tail(1), fc["week"]]),
                pd.concat([hist["sales"].tail(1), fc["demand_pred"]]), color="tab:red", marker="o", markersize=2.5,
                linewidth=1, label="forecast")
        ax.set_title(pid, fontsize=8); ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Weekly demand: history and forecast (+1 to +4 weeks)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write / validate outputs/demand_forecast.csv.")
    parser.add_argument("--check", action="store_true", help="only validate the existing file")
    args = parser.parse_args()
    config = dl.load_config()
    path = dl.PROJECT_ROOT / config["paths"]["outputs_dir"] / FORECAST_FILE
    try:
        if args.check:
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: run without --check first.")
            df = pd.read_csv(path)
            problems = validate_forecast(df, horizon=int(config["forecast"]["horizon"]))
            if problems:
                raise SystemExit("INVALID: " + "; ".join(problems))
            print(f"VALID: {df['product_id'].nunique()} products x {df['week'].nunique()} weeks, file {path}")
            return
        res = export_forecast(config)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}")
    info, fc = res["info"], res["forecast"]
    print(f"Saved {path}")
    print(f"VALID: {info['n_products']} products x {len(info['weeks'])} weeks = {info['n_rows']} rows | "
          f"origin {info['origin_week']} | seasonal_weight {info['seasonal_weight']} | {info['n_models']} model(s)")
    print("\nFirst rows:")
    print(fc.head(8).to_string(index=False))
    print("\nTotal forecast demand by week (all products):")
    for w, v in info["total_demand_by_week"].items():
        print(f"  {w}: {v:,.0f}")
    print("\nAlso saved: outputs/demand_forecast_details.csv, demand_forecast_info.json, plots/08_forecast_next_weeks.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
