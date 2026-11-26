"""Load and validate the four input files used by the pricing engine.

Each ``load_*`` function:
  1. reads one CSV,
  2. checks that the required columns exist (raises ``DataValidationError`` if not),
  3. cleans it (duplicates, bad values) WITHOUT touching the original file,
  4. returns a ``LoadResult`` = cleaned DataFrame + a list of human-readable warnings.

The original outputs of Agent 1 and Agent 2 are only read, never modified.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.pricing.config import load_config, resolve_path


class DataValidationError(Exception):
    """Raised when an input file is missing required columns or is unusable."""


@dataclass
class LoadResult:
    """A cleaned table together with everything we noticed while cleaning it."""
    data: pd.DataFrame
    warnings: list = field(default_factory=list)


# ---------- small helpers ----------

def _read_csv(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name}: file not found at {path}")
    return pd.read_csv(path)


def _require_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{name}: missing required columns {missing}. Found: {list(df.columns)}"
        )


def _drop_duplicates(df, subset, name, warnings):
    n = int(df.duplicated(subset=subset).sum())
    if n:
        warnings.append(f"{name}: removed {n} duplicated rows on {subset} (kept the last).")
        df = df.drop_duplicates(subset=subset, keep="last")
    return df


# ---------- Agent 1: demand forecast ----------

def load_forecast(config=None, path=None):
    """Load Agent 1's ``demand_forecast.csv`` (product_id, week, demand_pred)."""
    cfg = config or load_config()
    path = path or resolve_path(cfg["paths"]["forecast_file"])
    df = _read_csv(path, "demand_forecast")
    _require_columns(df, cfg["schema"]["forecast"], "demand_forecast")
    warnings = []

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df["demand_pred"] = pd.to_numeric(df["demand_pred"], errors="coerce")

    bad = int(df[["week", "demand_pred"]].isna().any(axis=1).sum())
    if bad:
        warnings.append(f"demand_forecast: dropped {bad} rows with invalid week or demand.")
        df = df.dropna(subset=["week", "demand_pred"])

    df = _drop_duplicates(df, ["product_id", "week"], "demand_forecast", warnings)

    neg = int((df["demand_pred"] < 0).sum())
    if neg:
        warnings.append(f"demand_forecast: {neg} negative forecasts clipped to 0.")
        df["demand_pred"] = df["demand_pred"].clip(lower=0)

    return LoadResult(df.reset_index(drop=True), warnings)


# ---------- Agent 2: reputation ----------

def load_reputation(config=None, path=None):
    """Load Agent 2's ``reputation_scores.csv``.

    The agreed interface was ``product_id, category, reputation_score, attributs``,
    but the real file is keyed by ``product_name``. We accept it and warn, instead of failing.
    """
    cfg = config or load_config()
    path = path or resolve_path(cfg["paths"]["reputation_file"])
    df = _read_csv(path, "reputation_scores")
    _require_columns(df, cfg["schema"]["reputation"], "reputation_scores")
    warnings = []

    for col in ("product_id", "category"):
        if col not in df.columns:
            warnings.append(
                f"reputation_scores: no '{col}' column (interface not respected by Agent 2) "
                "-> a mapping is needed to join with products."
            )

    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["reputation_score"] = pd.to_numeric(df["reputation_score"], errors="coerce")

    n_nan = int(df["reputation_score"].isna().sum())
    if n_nan:
        warnings.append(f"reputation_scores: dropped {n_nan} rows without a valid score.")
        df = df.dropna(subset=["reputation_score"])

    out_of_range = int((~df["reputation_score"].between(0, 1)).sum())
    if out_of_range:
        warnings.append(f"reputation_scores: {out_of_range} scores outside [0, 1] clipped.")
        df["reputation_score"] = df["reputation_score"].clip(0, 1)

    df = _drop_duplicates(df, ["product_name"], "reputation_scores", warnings)

    # Low-review products give unreliable averages: flag them, do not delete them.
    if "n_reviews" in df.columns:
        threshold = cfg["validation"]["min_reviews_reliable"]
        df["low_confidence"] = df["n_reviews"] < threshold
        n_low = int(df["low_confidence"].sum())
        if n_low:
            warnings.append(
                f"reputation_scores: {n_low}/{len(df)} products have fewer than "
                f"{threshold} reviews (flagged low_confidence)."
            )

    return LoadResult(df.reset_index(drop=True), warnings)


# ---------- Prices and costs ----------

def load_products(config=None, path=None):
    """Load product prices/costs and rename columns to the project standard
    (``current_price -> price_current``, ``product_cost -> cost``)."""
    cfg = config or load_config()
    path = path or resolve_path(cfg["paths"]["products_file"])
    df = _read_csv(path, "products").rename(columns=cfg["rename"]["products"])
    _require_columns(df, cfg["schema"]["products"], "products")
    warnings = []

    df["product_id"] = df["product_id"].astype(str).str.strip()
    for col in ("price_current", "cost", "competitor_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _drop_duplicates(df, ["product_id"], "products", warnings)

    # A product without a valid current price cannot be priced at all -> drop.
    bad_price = int((df["price_current"].isna() | (df["price_current"] <= 0)).sum())
    if bad_price:
        warnings.append(f"products: dropped {bad_price} rows with missing/non-positive price.")
        df = df[df["price_current"] > 0]

    # Invalid costs are KEPT but set to NaN: the constraints step (Phase 4) decides what to do.
    bad_cost = int((df["cost"].isna() | (df["cost"] <= 0)).sum())
    if bad_cost:
        warnings.append(f"products: {bad_cost} rows with missing/invalid cost (set to NaN).")
        df.loc[~(df["cost"] > 0), "cost"] = float("nan")

    bad_comp = int((df["competitor_price"].isna() | (df["competitor_price"] <= 0)).sum())
    if bad_comp:
        warnings.append(f"products: {bad_comp} rows with missing/invalid competitor_price (set to NaN).")
        df.loc[~(df["competitor_price"] > 0), "competitor_price"] = float("nan")

    return LoadResult(df.reset_index(drop=True), warnings)


# ---------- Agent 1: weekly sales history ----------

def load_sales_history(config=None, path=None):
    """Load Agent 1's ``sales_weekly.csv`` (history used later for the regression)."""
    cfg = config or load_config()
    path = path or resolve_path(cfg["paths"]["sales_file"])
    df = _read_csv(path, "sales_weekly")
    _require_columns(df, cfg["schema"]["sales"], "sales_weekly")
    warnings = []

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    bad = int(df[["week", "sales"]].isna().any(axis=1).sum())
    if bad:
        warnings.append(f"sales_weekly: dropped {bad} rows with invalid week or sales.")
        df = df.dropna(subset=["week", "sales"])

    df = _drop_duplicates(df, ["product_id", "week"], "sales_weekly", warnings)

    zero = int((df["sales"] == 0).sum())
    if zero:
        warnings.append(f"sales_weekly: {zero} weeks with zero sales (log(sales) needs care).")

    return LoadResult(df.sort_values(["product_id", "week"]).reset_index(drop=True), warnings)


# ---------- Run all four and print a report ----------

def load_all(config=None):
    """Load every input and return ``{name: LoadResult}``."""
    cfg = config or load_config()
    return {
        "forecast": load_forecast(cfg),
        "reputation": load_reputation(cfg),
        "products": load_products(cfg),
        "sales": load_sales_history(cfg),
    }


def print_report(results):
    """Print a short validation report for the demo / debugging."""
    for name, res in results.items():
        df = res.data
        print(f"\n=== {name}: {df.shape[0]} rows x {df.shape[1]} cols ===")
        print("columns:", list(df.columns))
        if res.warnings:
            for w in res.warnings:
                print("  WARNING:", w)
        else:
            print("  OK - no warnings")


if __name__ == "__main__":
    print_report(load_all())
