"""Price optimizer (Agent 3, Phase 4).

Demand model used for one product (constant-elasticity, the form estimated in regression.py):

    D(p) = D0 * (p / p0) ** e

    D0 = demand forecast by Agent 1 (expected demand at the CURRENT price p0)
    e  = price elasticity of this product (negative)

Profit model:   profit(p) = (p - cost) * D(p)

IMPORTANT HONESTY NOTE: the real data has no prices, so ``e`` is NOT measured. It starts from an
assumed base value (config: optimizer.base_elasticity) and is adjusted per product by three
documented assumptions: competitor price gap, reputation, and demand pressure. Results are
therefore SIMULATED recommendations, not proven gains.

Search: a price grid inside the allowed interval (constraints.py). For a constant elasticity
e < -1 the unconstrained optimum is also known analytically: p* = cost * e / (1 + e). We report it
to explain whether a constraint is limiting the recommendation.
"""
import math

import numpy as np
import pandas as pd

from src.pricing.config import load_config
from src.pricing.constraints import PriceConstraints, apply_constraints, price_bounds, resolve_cost


def _num(value, default=None):
    """Convert to float; return ``default`` for None / NaN."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


def product_elasticity(price_current, competitor_price, reputation, reputation_available,
                       demand_index, opt):
    """Elasticity of one product = base elasticity adjusted by three assumptions.

    |e| = |base| * (1 + gap_sens * gap) * (1 - rep_sens * (rep - rep_ref)) * (1 - dem_sens * (demand_index - 1))
      - gap  = price_current / competitor_price - 1   (pricier than the competitor -> more sensitive)
      - rep  = reputation score                        (better reputation -> less sensitive)
      - demand_index > 1                               (demand stronger than usual -> less sensitive)
    The result is kept between min_abs_elasticity and max_abs_elasticity.
    Returns (elasticity, dict of the factors used).
    """
    comp = _num(competitor_price)
    gap = price_current / comp - 1 if comp and comp > 0 else 0.0
    rep_diff = (reputation - opt["reputation_reference"]) if reputation_available and reputation is not None else 0.0
    di = demand_index if demand_index is not None else 1.0

    mult = ((1 + opt["gap_sensitivity"] * gap)
            * (1 - opt["reputation_sensitivity"] * rep_diff)
            * (1 - opt["demand_sensitivity"] * (di - 1)))
    abs_e = abs(opt["base_elasticity"]) * mult
    abs_e = min(max(abs_e, opt["min_abs_elasticity"]), opt["max_abs_elasticity"])
    return -abs_e, {"competitor_gap": gap, "reputation_diff": rep_diff, "demand_index": di,
                    "gap_known": comp is not None and comp > 0}


def expected_demand(price, price_current, demand_current, elasticity):
    """D(p) = D0 * (p / p0) ** e"""
    return demand_current * (price / price_current) ** elasticity


def expected_profit(price, cost, price_current, demand_current, elasticity):
    """profit(p) = (p - cost) * D(p)"""
    return (price - cost) * expected_demand(price, price_current, demand_current, elasticity)


def _binding(best, bounds):
    """Name the constraint that limits the chosen price (None if the optimum is inside the interval)."""
    if abs(best - bounds["hi"]) < 0.0051 and bounds["hi"] > bounds["lo"]:
        return bounds["hi_source"]
    if abs(best - bounds["lo"]) < 0.0051 and bounds["hi"] > bounds["lo"]:
        return bounds["lo_source"]
    return None


def _reasoning(r, factors, flags):
    """Plain-language explanation of the recommendation (for the API and the dashboard)."""
    parts = []
    pct = r["price_change_pct"]
    if r["status"] != "ok":
        return "No recommendation: " + ", ".join(flags) + ". The current price is kept."

    parts.append(
        f"Estimated price sensitivity (elasticity) is {r['elasticity_used']:.2f} "
        f"(base {r['base_elasticity']:.1f}; "
        + (f"priced {factors['competitor_gap']:+.1%} vs competitor, " if factors["gap_known"] else "no competitor price, ")
        + (f"reputation {factors['reputation_diff']:+.2f} vs typical, " if "reputation_unavailable" not in flags else "reputation unavailable (neutral), ")
        + f"demand index {factors['demand_index']:.2f}).")
    star = r["unconstrained_optimum"]
    if star is not None:
        parts.append(f"Without any limit the profit-maximizing price would be {star:.2f} "
                     f"({star / r['price_current'] - 1:+.0%} vs current).")
    if abs(pct) < 0.05:
        parts.append("The current price is already (almost) optimal, so it is kept.")
    else:
        parts.append(f"Moving the price {pct:+.1f}% changes demand by "
                     f"{r['demand_recommended'] / r['demand_current'] - 1:+.1%} and weekly profit by "
                     f"{r['profit_recommended'] / r['profit_current'] - 1:+.1%}.")
    if r["binding_constraint"]:
        parts.append(f"The change is limited by the rule '{r['binding_constraint']}'.")
    return " ".join(parts)


def optimal_price(product_data, cfg=None, last_price=None):
    """Recommend the profit-maximizing price for ONE product, respecting all constraints.

    Parameters
    ----------
    product_data : dict or pandas Series with
        product_id, price_current, cost, demand_pred   (required)
        competitor_price, reputation_score, reputation_available, demand_index  (optional)
    cfg : the loaded pricing_config.yaml (default: load it).
    last_price : previous recommended price (limits the change per update). Defaults to the current price.

    Returns a dict with: price_current, price_recommended, demand_current, demand_recommended,
    profit_current, profit_recommended, margin_gain, margin_pct_current/recommended, price_change_pct,
    elasticity_used, unconstrained_optimum, binding_constraint, status, flags, reasoning.
    "Profit" is per week in model units (forecast units x currency), useful for comparisons only.
    """
    cfg = cfg or load_config()
    rules, opt = PriceConstraints.from_config(cfg), cfg["optimizer"]
    get = product_data.get if hasattr(product_data, "get") else (lambda k, d=None: product_data[k])

    p0 = _num(get("price_current"))
    if p0 is None or p0 <= 0:
        raise ValueError(f"product {get('product_id')}: price_current must be a positive number.")
    d0 = _num(get("demand_pred"))
    flags = []
    out = {"product_id": get("product_id"), "price_current": p0, "base_elasticity": opt["base_elasticity"]}

    def no_rec(reason):
        out.update(price_recommended=p0, demand_current=d0, demand_recommended=d0, profit_current=None,
                   profit_recommended=None, margin_gain=0.0, margin_pct_current=None,
                   margin_pct_recommended=None, price_change_pct=0.0, elasticity_used=None,
                   unconstrained_optimum=None, binding_constraint=None, status="no_recommendation",
                   flags=flags + [reason])
        out["reasoning"] = _reasoning(out, {}, out["flags"])
        return out

    cost, cost_flag = resolve_cost(p0, _num(get("cost")), rules)
    if cost_flag:
        flags.append(cost_flag)
    if cost is None:
        return no_rec("missing_or_invalid_cost")
    if d0 is None or d0 < 0:
        return no_rec("missing_or_invalid_demand_forecast")

    rep_avail = bool(get("reputation_available", True))
    if not rep_avail:
        flags.append("reputation_unavailable")
    e, factors = product_elasticity(p0, get("competitor_price"), _num(get("reputation_score")), rep_avail,
                                    _num(get("demand_index"), 1.0), opt)
    if not factors["gap_known"]:
        flags.append("competitor_price_missing")

    bounds = price_bounds(p0, cost, rules, last_price)
    flags += bounds["flags"]

    # candidate prices: a regular grid + the current price + the analytic optimum (when inside bounds)
    star = cost * e / (1 + e) if e < -1 else None
    grid = list(np.linspace(bounds["lo"], bounds["hi"], opt["n_grid"]))
    for extra in (p0, star):
        if extra is not None and bounds["lo"] <= extra <= bounds["hi"]:
            grid.append(extra)
    grid = np.array(sorted(set(grid)))
    profits = expected_profit(grid, cost, p0, d0, e)
    best = float(grid[int(np.argmax(profits))])
    best, adj_flags = apply_constraints(best, p0, cost, rules, last_price)     # safety net + rounding
    flags += [f for f in adj_flags if f not in flags and f != "cost_missing_estimated_from_default_ratio"]

    d1 = expected_demand(best, p0, d0, e)
    prof0, prof1 = expected_profit(p0, cost, p0, d0, e), expected_profit(best, cost, p0, d0, e)
    out.update(
        price_recommended=best, demand_current=d0, demand_recommended=float(d1),
        profit_current=float(prof0), profit_recommended=float(prof1), margin_gain=float(prof1 - prof0),
        margin_pct_current=(p0 - cost) / p0, margin_pct_recommended=(best - cost) / best,
        price_change_pct=(best / p0 - 1) * 100, elasticity_used=float(e),
        unconstrained_optimum=float(star) if star else None,
        binding_constraint=_binding(best, bounds), status="ok", flags=flags)
    out["reasoning"] = _reasoning(out, factors, flags)
    return out


def recommend_all(master, cfg=None):
    """Run ``optimal_price`` for every product and week in the merged table.

    Weeks are processed in order and each recommendation becomes the ``last_price`` of the next
    one, so the price never moves by more than the allowed percentage between two updates.
    """
    cfg = cfg or load_config()
    rows = []
    for _, group in master.sort_values(["product_id", "week"]).groupby("product_id", sort=False):
        last = None
        for _, row in group.iterrows():
            res = optimal_price(row, cfg, last_price=last)
            res.update(week=row["week"], category=row.get("category"))
            last = res["price_recommended"]
            rows.append(res)
    df = pd.DataFrame(rows)
    df["flags"] = df["flags"].apply(lambda f: ";".join(f))
    return df


# ------------------------------------------------------------------------------------------
# Command-line summary:  python -m src.pricing.optimizer
# ------------------------------------------------------------------------------------------

def _summary(rec):
    ok = rec[rec["status"] == "ok"]
    return {"rows": len(rec), "raised_pct": (ok["price_change_pct"] > 0.5).mean() * 100,
            "lowered_pct": (ok["price_change_pct"] < -0.5).mean() * 100,
            "mean_change_pct": ok["price_change_pct"].mean(),
            "at_limit_pct": ok["binding_constraint"].notna().mean() * 100,
            "weekly_gain": ok["margin_gain"].sum(), "weekly_profit": ok["profit_current"].sum()}


if __name__ == "__main__":
    import copy

    from src.pipeline.integration import run_integration

    cfg = load_config()
    master = run_integration(cfg, save=False)[0].data
    rec = recommend_all(master, cfg)
    s = _summary(rec)
    print(f"{s['rows']} recommendations ({rec['product_id'].nunique()} products x {rec['week'].nunique()} weeks)")
    print(f"Price raised: {s['raised_pct']:.0f}% | lowered: {s['lowered_pct']:.0f}% | "
          f"mean change: {s['mean_change_pct']:+.2f}% | limited by a rule: {s['at_limit_pct']:.0f}%")
    print("Rules that limited the price:", rec["binding_constraint"].value_counts().to_dict())
    print(f"SIMULATED gain vs current prices: {s['weekly_gain']:,.0f} per week on a base profit of "
          f"{s['weekly_profit']:,.0f} ({s['weekly_gain'] / s['weekly_profit']:+.2%}) - model estimate, not real sales")

    print("\nSensitivity to the ASSUMED base elasticity (first week only):")
    first = master[master["week"] == master["week"].min()]
    for e in (-2.0, -2.5, -3.0, -4.0):
        c = copy.deepcopy(cfg)
        c["optimizer"]["base_elasticity"] = e
        s = _summary(recommend_all(first, c))
        print(f"  base elasticity {e:>5}: raised {s['raised_pct']:3.0f}% | lowered {s['lowered_pct']:3.0f}% | "
              f"mean change {s['mean_change_pct']:+5.1f}% | at a limit {s['at_limit_pct']:3.0f}% | "
              f"gain {s['weekly_gain'] / s['weekly_profit']:+.2%}")

    ex = rec[(rec["week"] == rec["week"].min()) & (rec["status"] == "ok")]
    ex = ex.loc[ex["price_change_pct"].abs().idxmax()]
    print(f"\nExample - product {ex['product_id']} ({ex['category']}), week {pd.Timestamp(ex['week']).date()}:")
    print(f"  price {ex['price_current']:.2f} -> {ex['price_recommended']:.2f} ({ex['price_change_pct']:+.1f}%)")
    print("  " + ex["reasoning"])
