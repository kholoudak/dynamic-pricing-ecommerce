"""Backtest: dynamic pricing versus the static (current) price  (Agent 3, Phase 5).

READ THIS FIRST - what this is and is not
-----------------------------------------
* The data contains NO real sales observed at different prices. A real historical backtest is
  therefore impossible, and nothing here is a real profit improvement.
* If we evaluated the recommendations with the same demand model that produced them, the result
  would be guaranteed to look good (the optimizer would just be agreeing with itself).
* So we do a SIMULATION that asks a harder question: "the optimizer ASSUMED an elasticity;
  if the real world is different, does the dynamic strategy still beat the static one?"

Strategies compared over the forecast weeks of every product:
  static  : keep the current price.
  dynamic : the recommended price of each week (recommend_all).

Simulated demand under a chosen TRUE elasticity e:   D = D0 * (p / p0) ** e
Three scenarios:
  A) "matched"  : true elasticity = the one the optimizer used (optimistic, circular - a reference).
  B) "grid"     : the same true elasticity for every product, from -1.5 to -5 (misspecification).
  C) "noisy"    : true elasticity = optimizer's value x random error per product (Monte Carlo).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.pricing.config import load_config, resolve_path
from src.pricing.optimizer import recommend_all

LABEL = "SIMULATED - no real sales outcomes at different prices"


def simulate(rec, elasticity):
    """Profit of both strategies for every row, given the TRUE elasticity (scalar or per-row array).

    static profit  = (p0 - cost) * D0                       (demand at the current price is D0)
    dynamic profit = (p1 - cost) * D0 * (p1 / p0) ** e
    """
    ok = rec[rec["status"] == "ok"].copy()
    e = np.broadcast_to(np.asarray(elasticity, dtype=float), (len(rec),))[(rec["status"] == "ok").to_numpy()]
    d0, p0, p1, c = ok["demand_current"], ok["price_current"], ok["price_recommended"], ok["cost"]
    ok["profit_static"] = (p0 - c) * d0
    ok["profit_dynamic"] = (p1 - c) * d0 * (p1 / p0) ** e
    ok["gain"] = ok["profit_dynamic"] - ok["profit_static"]
    return ok


def total_gain_pct(sim):
    """Total profit gain of dynamic versus static, in %."""
    return float(sim["gain"].sum() / sim["profit_static"].sum() * 100)


def concentration(sim, top_n=10):
    """How much of the gain comes from the `top_n` best products?

    Profit is weighted by demand volume, so a few large products can hide losses everywhere else.
    Returns the share of the net gain they provide and the gain of all the OTHER products.
    """
    per_product = sim.groupby("product_id")[["gain", "profit_static"]].sum().sort_values("gain", ascending=False)
    top, rest = per_product.head(top_n), per_product.iloc[top_n:]
    return {"top_n": top_n,
            "top_share_of_net_gain_pct": float(top["gain"].sum() / per_product["gain"].sum() * 100),
            "gain_excluding_top_pct": float(rest["gain"].sum() / rest["profit_static"].sum() * 100)}


def price_statistics(rec):
    """Average price level and price variation of the two strategies (no demand model needed)."""
    ok = rec[rec["status"] == "ok"].sort_values(["product_id", "week"])
    weekly_change = ok.groupby("product_id")["price_recommended"].apply(lambda s: s.pct_change().abs())
    first_step = (ok.groupby("product_id")["price_recommended"].first() /
                  ok.groupby("product_id")["price_current"].first() - 1).abs()
    return {
        "avg_price_static": float(ok["price_current"].mean()),
        "avg_price_dynamic": float(ok["price_recommended"].mean()),
        "avg_price_change_pct": float((ok["price_recommended"] / ok["price_current"] - 1).mean() * 100),
        "mean_abs_weekly_change_pct": float(pd.concat([first_step, weekly_change.dropna()]).mean() * 100),
        "price_cv_within_product_pct": float((ok.groupby("product_id")["price_recommended"].std() /
                                              ok.groupby("product_id")["price_recommended"].mean()).mean() * 100),
        "static_price_variation_pct": 0.0,
    }


def run_backtest(cfg=None, master=None, save=True):
    """Run the three simulated scenarios and return all results (and save tables + charts)."""
    cfg = cfg or load_config()
    bt = cfg["backtest"]
    if master is None:
        from src.pipeline.integration import run_integration
        master = run_integration(cfg, save=False)[0].data

    rec = recommend_all(master, cfg)
    ok_mask = (rec["status"] == "ok").to_numpy()

    # A) matched (circular reference)
    sim_a = simulate(rec, rec["elasticity_used"].fillna(-1.0).to_numpy())
    # B) one common true elasticity for all products
    grid = [{"true_elasticity": e, "gain_pct": total_gain_pct(simulate(rec, e)),
             "share_products_gaining_pct": float((simulate(rec, e).groupby("product_id")["gain"].sum() > 0).mean() * 100),
             **{k: v for k, v in concentration(simulate(rec, e)).items() if k != "top_n"}}
            for e in bt["true_elasticity_grid"]]
    # C) noisy product-level truth, Monte Carlo
    rng = np.random.default_rng(bt["seed"])
    products = rec["product_id"].unique()
    pid_index = pd.Series(range(len(products)), index=products)[rec["product_id"]].to_numpy()
    base_e = rec["elasticity_used"].fillna(-1.0).to_numpy()
    runs, last_sim = [], None
    cat_gain = []
    for _ in range(bt["n_runs"]):
        noise = np.exp(rng.normal(0, bt["noise_sd"], len(products)))[pid_index]
        sim = simulate(rec, base_e * noise)
        per_product = sim.groupby("product_id")["gain"].sum()
        runs.append({"gain_pct": total_gain_pct(sim), "share_products_gaining_pct": float((per_product > 0).mean() * 100)})
        cat_gain.append(sim.groupby("category")[["gain", "profit_static"]].sum())
        last_sim = sim
    mc = pd.DataFrame(runs)
    cat = sum(cat_gain) / len(cat_gain)
    cat["gain_pct"] = cat["gain"] / cat["profit_static"] * 100
    cat = cat.sort_values("gain_pct", ascending=False)

    results = {
        "label": LABEL,
        "n_products": int(rec["product_id"].nunique()), "n_weeks": int(rec["week"].nunique()),
        "assumed_base_elasticity": cfg["optimizer"]["base_elasticity"],
        "A_matched_gain_pct": total_gain_pct(sim_a),
        "B_grid": grid,
        "C_noisy": {"mean_gain_pct": float(mc["gain_pct"].mean()), "p5_gain_pct": float(mc["gain_pct"].quantile(0.05)),
                    "p95_gain_pct": float(mc["gain_pct"].quantile(0.95)),
                    "share_runs_positive_pct": float((mc["gain_pct"] > 0).mean() * 100),
                    "mean_share_products_gaining_pct": float(mc["share_products_gaining_pct"].mean())},
        "price_statistics": price_statistics(rec),
    }
    if save:
        out_dir = resolve_path(cfg["paths"]["reports_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.json_normalize(results).T.to_csv(out_dir / "backtest_summary.csv", header=["value"])
        cat.to_csv(out_dir / "backtest_by_category.csv")
        pd.DataFrame(grid).to_csv(out_dir / "backtest_elasticity_grid.csv", index=False)
        make_charts(rec, results, mc, cat, resolve_path(bt["figures_dir"]))
    return results, rec, mc, cat


def make_charts(rec, results, mc, cat, fig_dir):
    """Save 4 PNG charts. Every title says the results are simulated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    note = "SIMULATED (no real outcomes)"

    # 1. gain vs true elasticity - two bars each, so concentration cannot hide losses
    g = pd.DataFrame(results["B_grid"])
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, g["gain_pct"], 0.4, color="#2a9d8f", label="all products (weighted by volume)")
    ax.bar(x + 0.2, g["gain_excluding_top_pct"], 0.4, color="#e9c46a", label="all products EXCEPT the 10 largest gainers")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([str(e) for e in g["true_elasticity"]])
    ax.set_xlabel(f"TRUE elasticity of every product (optimizer assumed {results['assumed_base_elasticity']})")
    ax.set_ylabel("Profit gain vs static price (%)")
    ax.set_title(f"Does dynamic pricing survive a wrong assumption?\n{note}")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(fig_dir / "backtest_1_gain_vs_true_elasticity.png", dpi=130); plt.close(fig)

    # 2. Monte Carlo distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mc["gain_pct"], bins=25, color="#457b9d")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(mc["gain_pct"].mean(), color="#e76f51", ls="--", label=f"mean {mc['gain_pct'].mean():+.1f}%")
    ax.set_xlabel("Total profit gain vs static price (%)"); ax.set_ylabel("Monte Carlo runs")
    ax.set_title(f"Product-level elasticity errors ({len(mc)} runs)\n{note}")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "backtest_2_monte_carlo.png", dpi=130); plt.close(fig)

    # 3. by category
    fig, ax = plt.subplots(figsize=(8, 7))
    c = cat["gain_pct"].sort_values()
    ax.barh(c.index, c.values, color=["#2a9d8f" if v >= 0 else "#e76f51" for v in c.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Average profit gain vs static price (%)")
    ax.set_title(f"Performance by category (mean of Monte Carlo runs)\n{note}")
    fig.tight_layout(); fig.savefig(fig_dir / "backtest_3_by_category.png", dpi=130); plt.close(fig)

    # 4. price change distribution
    ok = rec[rec["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ok["price_change_pct"], bins=30, color="#8d99ae")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Recommended price vs current price (%)"); ax.set_ylabel("Product-weeks")
    ax.set_title("How far recommended prices move (model output, not an outcome)")
    fig.tight_layout(); fig.savefig(fig_dir / "backtest_4_price_changes.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    res, rec, mc, cat = run_backtest()
    print(f"=== BACKTEST - {LABEL} ===")
    print(f"{res['n_products']} products x {res['n_weeks']} weeks | optimizer assumed elasticity {res['assumed_base_elasticity']}\n")
    print(f"A) Matched world (circular reference, optimistic): {res['A_matched_gain_pct']:+.2f}%\n")
    print("B) If EVERY product's true elasticity were:")
    for r in res["B_grid"]:
        print(f"   {r['true_elasticity']:>5}: profit {r['gain_pct']:+6.2f}% | products gaining {r['share_products_gaining_pct']:3.0f}% | "
              f"top-10 products = {r['top_share_of_net_gain_pct']:.0f}% of the gain | all other products: {r['gain_excluding_top_pct']:+.1f}%")
    c = res["C_noisy"]
    print(f"\nC) Random product-level errors ({len(mc)} runs): mean {c['mean_gain_pct']:+.2f}% "
          f"(5%-95%: {c['p5_gain_pct']:+.2f}% to {c['p95_gain_pct']:+.2f}%), positive in {c['share_runs_positive_pct']:.0f}% of runs, "
          f"{c['mean_share_products_gaining_pct']:.0f}% of products gain on average")
    p = res["price_statistics"]
    print(f"\nPrices: average static {p['avg_price_static']:.2f} vs dynamic {p['avg_price_dynamic']:.2f} "
          f"({p['avg_price_change_pct']:+.2f}% on average)")
    print(f"Variation: mean absolute change per weekly update {p['mean_abs_weekly_change_pct']:.1f}% "
          f"(static: 0%); within-product price CV {p['price_cv_within_product_pct']:.1f}%")
    print("\nBest / worst categories (mean gain %):")
    print(" ", cat["gain_pct"].head(3).round(1).to_dict())
    print(" ", cat["gain_pct"].tail(3).round(1).to_dict())
    print("\nSaved tables to reports/ and charts to reports/figures/")
