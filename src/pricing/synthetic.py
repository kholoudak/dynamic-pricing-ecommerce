"""SIMULATED price experiment (Phase 3, part C).

WHY: the real sales data has no prices, so a real price elasticity cannot be measured.
To show that the regression method WORKS, we build a fake history where we KNOW the true
elasticity, then check that the regression finds it again.

Everything produced here carries ``is_synthetic = True``. Never present it as real results.

Demand rule used to generate the data (all in logs):
  log(demand) = log(baseline_demand)
              + elasticity            * log(price / usual_price)
              + competitor_effect     * log(competitor_price / usual_competitor_price)
              + reputation_effect     * (reputation - average reputation)
              + random noise
"""
import numpy as np
import pandas as pd


def make_synthetic_panel(master, syn_cfg):
    """Return a product x week table with simulated prices and demand.

    Parameters
    ----------
    master : DataFrame from integration (needs product_id, category, price_current,
             competitor_price, reputation_score, demand_pred).
    syn_cfg : the ``regression.synthetic`` block of pricing_config.yaml.
    """
    rng = np.random.default_rng(syn_cfg["seed"])

    base = master.drop_duplicates("product_id")[
        ["product_id", "category", "price_current", "competitor_price", "reputation_score"]
    ]
    baseline = master.groupby("product_id")["demand_pred"].mean().rename("baseline_demand")
    base = base.merge(baseline, on="product_id")

    weeks = pd.date_range("2015-08-17", periods=syn_cfg["n_weeks"], freq="W-MON")
    panel = base.merge(pd.DataFrame({"week": weeks}), how="cross")
    n = len(panel)

    # random weekly price moves; the competitor partly follows us (this creates collinearity)
    dev_own = rng.normal(0, syn_cfg["price_noise_sd"], n)
    dev_comp = syn_cfg["competitor_follow"] * dev_own + rng.normal(0, syn_cfg["competitor_noise_sd"], n)

    panel["price"] = panel["price_current"] * np.exp(dev_own)
    panel["competitor_price_t"] = panel["competitor_price"] * np.exp(dev_comp)

    rep_centered = panel["reputation_score"] - panel["reputation_score"].mean()
    log_demand = (
        np.log(panel["baseline_demand"])
        + syn_cfg["true_elasticity"] * dev_own
        + syn_cfg["true_competitor_effect"] * dev_comp
        + syn_cfg["true_reputation_effect"] * rep_centered
        + rng.normal(0, syn_cfg["demand_noise_sd"], n)
    )
    panel["demand"] = np.exp(log_demand)
    panel["is_synthetic"] = True

    cols = ["product_id", "category", "week", "price", "competitor_price_t", "reputation_score",
            "baseline_demand", "demand", "is_synthetic"]
    return panel[cols].sort_values(["week", "product_id"]).reset_index(drop=True)
