"""
Step 4 - Cleaning and weekly aggregation / Etape 4 - Nettoyage et agregation hebdomadaire.

Pipeline (every decision is documented in the docstring of the function that applies it):
    train.csv (3 M rows: day x store x family)
      1. aggregate_daily            -> one row per day x product (sales summed over the 54 stores)
      2. compute_open_stores        -> number of stores that sold something each day
      3. fill_missing_calendar_days -> the 25 Dec of each year is absent: inserted with sales = 0
      4. to_weekly + drop_partial_weeks -> Monday-based weeks, incomplete weeks removed
      5. trim_leading_zeros         -> weeks before the first sale of a product are removed
      6. history_summary / select_products -> products with a too-short history are excluded
      7. flag_outliers              -> extreme weeks are FLAGGED, never modified or deleted

Result: outputs/sales_weekly.csv  (columns: product_id, week, sales, onpromotion, open_stores, is_outlier)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.forecast import data_loader as dl

logger = logging.getLogger(__name__)

DEFAULTS = {
    "min_history_weeks": 104,   # a product needs at least ~2 years of weekly history
    "outlier_window": 9,        # centred rolling-median window (weeks)
    "outlier_k": 6.0,           # flag if |x - rolling median| > k * robust sigma
    "weekly_file": "sales_weekly.csv",
}


def get_params(config: dict) -> dict:
    """Return the `preprocessing` settings of config.yaml, completed with defaults."""
    return {**DEFAULTS, **(config.get("preprocessing") or {})}


# --------------------------------------------------------------------------- diagnostics
def zero_diagnostics(train: pd.DataFrame, config: dict) -> dict:
    """Test the hypothesis "many zeros come from stores that were closed / not yet open".

    A (date, store) is "closed" when the store sold NOTHING in any of the 33 families that day.
    Output: dict with the share of zero rows, the share of zero rows explained by closed
    store-days, and the number of closed store-days per store (top 10).
    """
    d = config["data"]
    key = [d["date_col"], "store_nbr"]
    store_total = train.groupby(key, observed=True)[d["target_col"]].transform("sum")
    is_zero = train[d["target_col"]] == 0
    closed_row = store_total == 0
    closed_days = (
        train.loc[closed_row].drop_duplicates(subset=key).groupby("store_nbr", observed=True).size()
        .sort_values(ascending=False)
    )
    return {
        "zero_share_pct": float(is_zero.mean() * 100),
        "zero_rows_explained_by_closed_store_days_pct": float((is_zero & closed_row).sum() / max(is_zero.sum(), 1) * 100),
        "closed_store_days": int(closed_days.sum()),
        "stores_with_closed_days": int(len(closed_days)),
        "top_stores_closed_days": closed_days.head(10).to_dict(),
    }


# --------------------------------------------------------------------------- step 1-3
def aggregate_daily(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Sum sales and promotions over all stores: one row per (date, product).

    Decision: product_id = `family` (the dataset has no product identifier) and the series
    is the national total, which keeps 33 light series (CPU friendly).
    Output columns: date, product_id, sales, onpromotion.
    """
    d = config["data"]
    daily = (
        train.groupby([d["date_col"], d["product_col"]], observed=True)
        .agg(sales=(d["target_col"], "sum"), onpromotion=(d["promo_col"], "sum"))
        .reset_index()
        .rename(columns={d["date_col"]: "date", d["product_col"]: "product_id"})
    )
    daily["product_id"] = daily["product_id"].astype(str).str.strip()
    return daily


def compute_open_stores(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Count, for each day, the stores with strictly positive total sales.

    Why: stores open progressively and close for some days; this changes the level of the
    national total independently of the demand per store. Output columns: date, open_stores.
    """
    d = config["data"]
    store_day = train.groupby([d["date_col"], "store_nbr"], observed=True)[d["target_col"]].sum()
    open_stores = (store_day > 0).groupby(level=0).sum().astype(int)
    return open_stores.rename("open_stores").rename_axis("date").reset_index()


def fill_missing_calendar_days(daily: pd.DataFrame, open_stores: pd.DataFrame) -> pd.DataFrame:
    """Insert the calendar days absent from the data (25 Dec) for every product.

    Decision: the day is inserted with sales = 0, onpromotion = 0, open_stores = 0 (stores are
    assumed closed: the whole day is absent from the file, for all stores and families).
    The weekly total is NOT rescaled. A boolean column `is_imputed_day` keeps track of it.
    """
    products = sorted(daily["product_id"].unique())
    full_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    index = pd.MultiIndex.from_product([full_days, products], names=["date", "product_id"])
    out = daily.set_index(["date", "product_id"]).reindex(index)
    out["is_imputed_day"] = out["sales"].isna()
    out["sales"] = out["sales"].fillna(0.0)
    out["onpromotion"] = out["onpromotion"].fillna(0)
    out = out.reset_index().merge(open_stores, on="date", how="left")
    out["open_stores"] = out["open_stores"].fillna(0).astype(int)
    return out


# --------------------------------------------------------------------------- step 4-5
def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate days into weeks starting on Monday (`week` = date of that Monday).

    sales / onpromotion are summed, open_stores is averaged, n_days counts the days present.
    """
    df = daily.copy()
    df["week"] = df["date"] - pd.to_timedelta(df["date"].dt.day_of_week, unit="D")
    weekly = (
        df.groupby(["product_id", "week"])
        .agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"),
             open_stores=("open_stores", "mean"), n_days=("date", "nunique"),
             n_imputed_days=("is_imputed_day", "sum"))
        .reset_index()
    )
    return weekly


def drop_partial_weeks(weekly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove weeks with fewer than 7 days (a partial week looks like a false drop in demand).

    With this dataset: the first week (starts on Tuesday 2013-01-01) and the last one
    (ends on Tuesday 2017-08-15). Output: (complete weeks, removed weeks).
    """
    complete = weekly["n_days"] == 7
    return weekly[complete].reset_index(drop=True), weekly[~complete].reset_index(drop=True)


def trim_leading_zeros(weekly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove, for each product, the weeks BEFORE its first week with sales > 0.

    Reason: those zeros mean "not sold / not recorded yet", not "zero demand". Zeros inside
    the series are kept (they are real). Products that never sell are dropped.
    Output: (trimmed table, number of removed weeks per product).
    """
    first = weekly.loc[weekly["sales"] > 0].groupby("product_id")["week"].min().rename("first_sale_week")
    merged = weekly.merge(first, on="product_id", how="left")
    keep = merged["week"] >= merged["first_sale_week"]
    removed = (~keep).groupby(merged["product_id"]).sum().astype(int).rename("leading_zero_weeks_removed")
    return merged.loc[keep].drop(columns="first_sale_week").reset_index(drop=True), removed


# --------------------------------------------------------------------------- step 6-7
def history_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    """One row per product: number of weeks, first/last week, zero weeks, mean weekly sales."""
    g = weekly.groupby("product_id")
    summary = pd.DataFrame({
        "n_weeks": g.size(),
        "first_week": g["week"].min(),
        "last_week": g["week"].max(),
        "zero_weeks": g["sales"].apply(lambda s: int((s == 0).sum())),
        "mean_weekly_sales": g["sales"].mean().round(1),
    })
    summary["zero_weeks_pct"] = (summary["zero_weeks"] / summary["n_weeks"] * 100).round(1)
    return summary.sort_values("n_weeks")


def select_products(summary: pd.DataFrame, min_weeks: int) -> Tuple[list, list]:
    """Split products into (kept, excluded) according to the minimum weekly history."""
    kept = summary.index[summary["n_weeks"] >= min_weeks].tolist()
    excluded = summary.index[summary["n_weeks"] < min_weeks].tolist()
    return kept, excluded


def flag_outliers(weekly: pd.DataFrame, window: int = 9, k: float = 6.0) -> pd.Series:
    """Flag extreme weeks WITHOUT changing the data (robust, per product).

    A week is flagged when |sales - centred rolling median| > k * sigma, with
    sigma = max(1.4826 * MAD of the residuals, 5 % of the typical level).
    The rolling median follows trend and seasonality, so normal peaks are not flagged.
    Only weeks with a FULL window are tested (the first/last window//2 weeks are never flagged):
    a truncated window is biased on a trending series and would create false alarms.
    The flag is only for inspection: values are kept as real sales.
    Output: boolean Series aligned with `weekly`.
    """
    flags = pd.Series(False, index=weekly.index)
    for _, g in weekly.sort_values("week").groupby("product_id"):
        med = g["sales"].rolling(window, center=True, min_periods=window).median()
        resid = g["sales"] - med
        mad = resid.abs().median()
        # floor: the scale is never below 5 % of the typical level, otherwise a nearly
        # perfect series (MAD ~ 0) would flag ordinary fluctuations
        sigma = max(1.4826 * mad, 0.05 * float(np.nanmedian(med.abs()))) if med.notna().any() else np.nan
        if not np.isfinite(sigma) or sigma == 0:
            continue
        flags.loc[g.index] = (resid.abs() > k * sigma).fillna(False)
    return flags


# --------------------------------------------------------------------------- pipeline
def run_preprocessing(config: Optional[dict] = None, train: Optional[pd.DataFrame] = None,
                      out_dir: Optional[Path] = None, processed_dir: Optional[Path] = None) -> dict:
    """Run the whole Step 4 pipeline and write the files.

    Writes: outputs/sales_weekly.csv, outputs/excluded_products.csv,
            outputs/data_quality_report.csv, data/processed/daily_by_product.csv
    Input : config (default config.yaml); train (default: loaded from data/raw).
    Output: dict with the weekly table, the summary and the counters of every cleaning step.
    """
    config = config or dl.load_config()
    p = get_params(config)
    if train is None:
        train = dl.load_train(config)
    out_dir = Path(out_dir) if out_dir else dl.PROJECT_ROOT / config["paths"]["outputs_dir"]
    processed_dir = Path(processed_dir) if processed_dir else dl.PROJECT_ROOT / config["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    daily = aggregate_daily(train, config)
    days_before = daily["date"].nunique()
    daily = fill_missing_calendar_days(daily, compute_open_stores(train, config))
    imputed_days = int(daily.loc[daily["is_imputed_day"], "date"].nunique())
    weekly = to_weekly(daily)
    weekly, partial = drop_partial_weeks(weekly)
    all_products = set(weekly["product_id"])
    weekly, leading = trim_leading_zeros(weekly)
    never_sold = sorted(all_products - set(weekly["product_id"]))
    summary = history_summary(weekly)
    kept, excluded = select_products(summary, p["min_history_weeks"])
    weekly = weekly[weekly["product_id"].isin(kept)].reset_index(drop=True)
    weekly["is_outlier"] = flag_outliers(weekly, p["outlier_window"], p["outlier_k"])

    final = weekly[["product_id", "week", "sales", "onpromotion", "open_stores", "is_outlier"]]
    final = final.sort_values(["product_id", "week"]).reset_index(drop=True)
    final.to_csv(out_dir / p["weekly_file"], index=False)
    summary.assign(excluded=summary.index.isin(excluded), leading_zero_weeks_removed=leading,
                   outlier_weeks=final.groupby("product_id")["is_outlier"].sum()
                   ).to_csv(out_dir / "data_quality_report.csv")
    summary.loc[excluded].to_csv(out_dir / "excluded_products.csv")
    daily.to_csv(processed_dir / "daily_by_product.csv", index=False)

    report = {
        "days_in_file": int(days_before), "days_inserted": imputed_days,
        "partial_weeks_removed": int(partial["week"].nunique()),
        "leading_zero_weeks_removed": int(leading.sum()),
        "products_kept": kept, "products_excluded": excluded, "products_never_sold": never_sold,
        "weeks_in_output": int(len(final)), "outlier_weeks": int(final["is_outlier"].sum()),
    }
    logger.info("Preprocessing report: %s", report)
    return {"weekly": final, "summary": summary, "report": report}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_preprocessing()
    for key, value in result["report"].items():
        print(f"{key}: {value}")
