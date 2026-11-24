"""End-to-end pricing pipeline (Agent 3, Phase 6).

    1. load + validate the inputs (Agent 1 forecast, Agent 2 reputation, prices/costs, sales history)
    2. build the product mapping and merge everything into the master table
    3. run the elasticity regression (validation on SIMULATED data - see regression.py)
    4. compute the optimal price of every product and week (optimizer.py)
    5. apply the pricing constraints (done inside the optimizer) and AUDIT the result
    6. export data/processed/recommended_prices.csv (+ pipeline_run.json with the run details)

Mock data -> real data: nothing in this file changes. Every input path comes from
pricing_config.yaml, so swapping a mock file for the real output of Agent 1 or 2 only means
changing (or overwriting) that file.

Run it:   python -m src.pipeline.pipeline
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.pipeline.integration import run_integration
from src.pricing.config import load_config, resolve_path
from src.pricing.constraints import PriceConstraints
from src.pricing.optimizer import recommend_all

SIMULATION_NOTE = ("Recommendations come from a model with an ASSUMED elasticity (the data has no prices). "
                   "Profit and gain figures are simulated estimates, not observed results.")

# The first five columns are the interface agreed in the project plan.
OUTPUT_COLUMNS = [
    "product_id", "week", "price_current", "price_recommended", "margin_gain",
    "category", "cost", "competitor_price", "reputation_score", "reputation_available", "demand_index",
    "demand_current", "demand_recommended", "profit_current", "profit_recommended",
    "margin_pct_current", "margin_pct_recommended", "price_change_pct", "elasticity_used",
    "unconstrained_optimum", "binding_constraint", "status", "flags", "reasoning",
]


class PipelineError(Exception):
    """Raised when the pipeline cannot produce trustworthy recommendations."""


@dataclass
class PipelineResult:
    recommendations: pd.DataFrame
    metadata: dict
    warnings: list = field(default_factory=list)


def audit_recommendations(rec, cfg):
    """Re-check every recommendation against the rules. Returns a list of violations (empty = OK).

    This is an independent safety net: the optimizer already enforces the rules, but a bug there
    must never silently reach the CSV, the API or the dashboard.
    """
    rules = PriceConstraints.from_config(cfg)
    ok = rec[rec["status"] == "ok"]
    problems = []
    tol = 1e-6

    if ok["price_recommended"].isna().any() or (ok["price_recommended"] <= 0).any():
        problems.append("non-positive or missing recommended price")

    margin_ok = ok["price_recommended"] >= ok["cost"] * (1 + rules.min_margin_pct) * (1 - tol)
    if (~margin_ok).any():
        problems.append(f"{int((~margin_ok).sum())} rows below the minimum margin")

    overridden = ok["flags"].str.contains("overrides_price_ceiling")
    too_high = ok["price_recommended"] > ok["price_current"] * rules.ceiling_pct_of_current * (1 + tol)
    too_low = ok["price_recommended"] < ok["price_current"] * rules.floor_pct_of_current * (1 - tol)
    if (too_high & ~overridden).any():
        problems.append(f"{int((too_high & ~overridden).sum())} rows above the price ceiling")
    if too_low.any():
        problems.append(f"{int(too_low.sum())} rows below the price floor")

    ordered = ok.sort_values(["product_id", "week"])
    previous = ordered.groupby("product_id")["price_recommended"].shift(1).fillna(ordered["price_current"])
    step = (ordered["price_recommended"] / previous - 1).abs()
    relaxed = ordered["flags"].str.contains("relaxed|overrides")
    too_big = (step > rules.max_change_per_update_pct + tol) & ~relaxed
    if too_big.any():
        problems.append(f"{int(too_big.sum())} rows change the price by more than "
                        f"{rules.max_change_per_update_pct:.0%} in one update")
    return problems


def run_pipeline(cfg=None, rebuild_mapping=False, run_regression=True, save=True, log=None):
    """Run the whole pipeline and return a PipelineResult.

    Parameters
    ----------
    cfg : loaded pricing_config.yaml (default: load it). Tests pass a config pointing at mock files.
    rebuild_mapping : rebuild product_mapping.csv instead of re-using the existing file.
    run_regression : also run the elasticity validation (adds ~2 s; stored in the run details).
    save : write recommended_prices.csv and pipeline_run.json.
    log : optional function that receives progress messages (e.g. ``print``).
    """
    cfg = cfg or load_config()
    log = log or (lambda msg: None)
    started = time.time()
    warnings, steps = [], []

    def step(name, t0):
        steps.append({"step": name, "seconds": round(time.time() - t0, 2)})
        log(f"[ok] {name} ({steps[-1]['seconds']}s)")

    # 1-2. load, validate, map and merge
    t0 = time.time()
    master_res, integ_warnings = run_integration(cfg, rebuild_mapping=rebuild_mapping, save=save)
    master = master_res.data
    warnings += integ_warnings
    step(f"load + validate + merge inputs -> {len(master)} product-week rows", t0)

    # 3. regression (validation only: the real data has no prices, see regression.py)
    regression_summary = None
    if run_regression:
        t0 = time.time()
        try:
            from src.pricing.regression import run_regression_report
            results, _ = run_regression_report(cfg, save=save, master=master)
            em, pm = results["elasticity_model"], results["promotion_model"]
            regression_summary = {
                "simulated_true_elasticity": em["true_elasticity"],
                "simulated_estimated_elasticity": em["estimated_elasticity"],
                "model_used": em["model_used"],
                "holdout_R2_vs_baseline": em["holdout_metrics"].get("R2_vs_baseline"),
                "real_data_promo_coefficient": pm["promo_coefficient"],
                "real_data_promo_test_R2": pm["test_R2_log"],
            }
            step("elasticity regression (validated on simulated data)", t0)
        except Exception as exc:                       # validation must not block pricing, but is never silent
            warnings.append(f"regression step failed: {exc}")
            log(f"[!!] regression step failed: {exc}")

    # 4-5. optimize + constraints
    t0 = time.time()
    rec = recommend_all(master, cfg)
    extra = master[["product_id", "week", "competitor_price", "reputation_score",
                    "reputation_available", "demand_index"]]
    rec = rec.merge(extra, on=["product_id", "week"], how="left")
    rec = rec[OUTPUT_COLUMNS].sort_values(["product_id", "week"]).reset_index(drop=True)
    step("optimal prices with constraints", t0)

    n_no_rec = int((rec["status"] != "ok").sum())
    if n_no_rec:
        warnings.append(f"{n_no_rec} product-weeks without recommendation (current price kept).")

    # audit
    t0 = time.time()
    problems = audit_recommendations(rec, cfg)
    if problems:
        raise PipelineError("Constraint audit failed: " + "; ".join(problems))
    step("constraint audit passed", t0)

    ok = rec[rec["status"] == "ok"]
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": SIMULATION_NOTE,
        "n_products": int(rec["product_id"].nunique()),
        "n_weeks": int(rec["week"].nunique()),
        "n_rows": int(len(rec)),
        "n_no_recommendation": n_no_rec,
        "assumed_base_elasticity": cfg["optimizer"]["base_elasticity"],
        "simulated_weekly_gain": float(ok["margin_gain"].sum()),
        "simulated_weekly_profit_current": float(ok["profit_current"].sum()),
        "simulated_gain_pct": float(ok["margin_gain"].sum() / ok["profit_current"].sum() * 100),
        "mean_price_change_pct": float(ok["price_change_pct"].mean()),
        "regression": regression_summary,
        "steps": steps,
        "warnings": warnings,
        "duration_seconds": round(time.time() - started, 2),
    }

    if save:
        out = resolve_path(cfg["paths"]["recommendations_file"])
        out.parent.mkdir(parents=True, exist_ok=True)
        rec.to_csv(out, index=False)
        (out.parent / "pipeline_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        log(f"[ok] exported {len(rec)} rows to {out.name}")
    return PipelineResult(rec, metadata, warnings)


def load_recommendations(cfg=None):
    """Read the exported recommended_prices.csv (used by the API and the dashboard)."""
    cfg = cfg or load_config()
    path = resolve_path(cfg["paths"]["recommendations_file"])
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run the pipeline first: python -m src.pipeline.pipeline")
    df = pd.read_csv(path, dtype={"product_id": str})
    df["week"] = pd.to_datetime(df["week"])
    df["flags"] = df["flags"].fillna("")
    return df


def load_run_info(cfg=None):
    """Read pipeline_run.json (timestamp, warnings, simulated totals). Returns {} if missing."""
    cfg = cfg or load_config()
    path = resolve_path(cfg["paths"]["recommendations_file"]).parent / "pipeline_run.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


if __name__ == "__main__":
    import sys

    result = run_pipeline(rebuild_mapping="--rebuild-mapping" in sys.argv, log=print)
    m = result.metadata
    print(f"\nDone in {m['duration_seconds']}s: {m['n_products']} products x {m['n_weeks']} weeks = {m['n_rows']} recommendations")
    print(f"Mean price change {m['mean_price_change_pct']:+.2f}% | simulated gain {m['simulated_gain_pct']:+.2f}% (model estimate)")
    print("NOTE:", m["note"])
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(" -", w)
