"""Merge Agent 1 (demand), Agent 2 (reputation) and the price/cost file into ONE table.

Grain of the result: one row per (product, forecast week).

Important modelling choice (documented in the report):
  Agent 1 forecasts demand for a whole FAMILY. We split it equally between the products that
  belong to that family: demand_pred = family demand / number of products in the family.
  This is a simplification: it only affects the SCALE of demand, not the price logic.
"""
import pandas as pd

from src.pipeline.loaders import DataValidationError, LoadResult, load_all
from src.pipeline.mapping import get_mapping
from src.pricing.config import load_config, resolve_path


def build_master_table(forecast, reputation, products, mapping, required_columns,
                       sales=None, recent_weeks=12):
    """Return a LoadResult with the merged table and a list of warnings.

    If ``sales`` (Agent 1's weekly history) is given, we also compute ``demand_index`` =
    forecast demand / average demand of the last ``recent_weeks`` weeks of the same family.
    An index above 1 means "demand is expected to be stronger than recently"."""
    warnings = []

    # 1) products + mapping -> every product gets a category and a reputation name
    master = products.merge(mapping, on="product_id", how="left", validate="one_to_one")
    master["n_products_in_category"] = master.groupby("category")["product_id"].transform("count")

    # 2) add the forecast of the product's family (one row per week)
    fc = forecast.rename(columns={"product_id": "category", "demand_pred": "demand_pred_family"})
    master = master.merge(fc, on="category", how="left", validate="many_to_many")
    if master["demand_pred_family"].isna().any():
        raise DataValidationError("integration: some products have no demand forecast.")
    master["demand_pred"] = master["demand_pred_family"] / master["n_products_in_category"]

    master["demand_index"] = 1.0
    if sales is not None and len(sales):
        cutoff = sales["week"].max() - pd.Timedelta(weeks=recent_weeks)
        recent = sales[sales["week"] > cutoff].groupby("product_id")["sales"].mean()
        index = master["demand_pred_family"] / master["category"].map(recent)
        index = index.replace([float("inf"), float("-inf")], float("nan"))
        n_bad = int(index.isna().sum())
        if n_bad:
            warnings.append(f"integration: demand_index unavailable for {n_bad} rows (set to 1.0).")
        master["demand_index"] = index.fillna(1.0)

    # 3) add reputation (joined through the mapping, by name)
    rep_cols = [c for c in ["product_name", "reputation_score", "n_reviews", "low_confidence",
                            "positive_ratio", "negative_ratio"] if c in reputation.columns]
    rep = reputation[rep_cols].rename(columns={"product_name": "reputation_name"})
    master = master.merge(rep, on="reputation_name", how="left")
    master["reputation_available"] = master["reputation_score"].notna()
    n_missing = int((~master.drop_duplicates("product_id")["reputation_available"]).sum())
    if n_missing:
        median = master["reputation_score"].median()
        master["reputation_score"] = master["reputation_score"].fillna(median)
        warnings.append(f"integration: {n_missing} products have no reputation; filled with the "
                        f"median ({median:.3f}) and flagged reputation_available=False.")
    if "low_confidence" in master.columns:
        master["low_confidence"] = master["low_confidence"].astype("boolean").fillna(False).astype(bool)

    # 4) simple derived columns used later as explanations / controls
    master["price_ratio"] = master["price_current"] / master["competitor_price"]
    master["margin_current_pct"] = (master["price_current"] - master["cost"]) / master["price_current"]

    # 5) final checks
    missing = [c for c in required_columns if c not in master.columns]
    if missing:
        raise DataValidationError(f"integration: required columns missing: {missing}")
    dup = int(master.duplicated(["product_id", "week"]).sum())
    if dup:
        raise DataValidationError(f"integration: {dup} duplicated (product_id, week) rows.")

    master = master.sort_values(["product_id", "week"]).reset_index(drop=True)
    return LoadResult(master, warnings)


def run_integration(cfg=None, rebuild_mapping=False, save=True):
    """Load all inputs, build the mapping, merge, and (optionally) save master_table.csv.

    Returns (master LoadResult, all warnings from every step)."""
    cfg = cfg or load_config()
    inputs = load_all(cfg)
    all_warnings = [w for r in inputs.values() for w in r.warnings]

    mapping, msg = get_mapping(cfg, inputs["products"].data, inputs["forecast"].data,
                               inputs["reputation"].data, rebuild=rebuild_mapping)
    all_warnings.append(msg)

    result = build_master_table(inputs["forecast"].data, inputs["reputation"].data,
                                inputs["products"].data, mapping, cfg["schema"]["master"],
                                sales=inputs["sales"].data,
                                recent_weeks=cfg.get("optimizer", {}).get("demand_index_weeks", 12))
    all_warnings += result.warnings

    if save:
        out = resolve_path(cfg["paths"]["master_file"])
        out.parent.mkdir(parents=True, exist_ok=True)
        result.data.to_csv(out, index=False)
        all_warnings.append(f"integration: saved {len(result.data)} rows to {out.name}.")
    return result, all_warnings


if __name__ == "__main__":
    import sys
    res, warns = run_integration(rebuild_mapping="--rebuild" in sys.argv)
    df = res.data
    print(f"Master table: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Products: {df['product_id'].nunique()} | Categories: {df['category'].nunique()} | "
          f"Weeks: {df['week'].nunique()}")
    print("\nNotes:")
    for w in warns:
        print(" -", w)
    print("\nSample:")
    print(df[["product_id", "category", "week", "demand_pred", "price_current", "cost",
              "competitor_price", "reputation_score", "reputation_available"]].head(6).to_string(index=False))
