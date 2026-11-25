"""Tests for src/pricing/synthetic.py and src/pricing/regression.py (Agent 3, Phase 3)."""
import numpy as np
import pandas as pd
import pytest

from src.pricing.config import load_config
from src.pricing.regression import (
    add_features, chronological_split, fit_elasticity_model, fit_promo_effect,
)
from src.pricing.synthetic import make_synthetic_panel

REG = load_config()["regression"]


@pytest.fixture(scope="module")
def master():
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame({
        "product_id": [str(i) for i in range(n)], "category": ["A"] * n,
        "price_current": rng.uniform(10, 300, n), "competitor_price": rng.uniform(10, 300, n),
        "reputation_score": rng.uniform(0.4, 0.9, n), "demand_pred": rng.uniform(100, 5000, n),
    })


@pytest.fixture(scope="module")
def panel(master):
    return make_synthetic_panel(master, REG["synthetic"])


def test_synthetic_panel_is_labelled_and_complete(panel):
    assert panel["is_synthetic"].all()
    assert len(panel) == 30 * REG["synthetic"]["n_weeks"]
    assert (panel["price"] > 0).all() and (panel["demand"] > 0).all()


def test_split_is_chronological(panel):
    train, test = chronological_split(panel, 0.3)
    assert train["week"].max() < test["week"].min()          # no future data in training
    assert abs(test["week"].nunique() / panel["week"].nunique() - 0.3) < 0.05


def test_elasticity_is_recovered(panel):
    res = fit_elasticity_model(panel, REG)
    assert abs(res.elasticity - REG["synthetic"]["true_elasticity"]) < 0.15
    assert res.elasticity < 0                                 # demand falls when price rises
    assert res.holdout_metrics["R2_vs_baseline"] > 0.3        # beats "ignore price"
    assert len(res.cv_table) == REG["n_cv_splits"]


def test_ols_used_when_no_collinearity(panel):
    res = fit_elasticity_model(panel, REG)
    assert not res.used_ridge and res.model.kind == "OLS"
    assert res.model.p_values["log_price_rel"] < 0.05


def test_ridge_used_when_collinear(master):
    cfg = {**REG["synthetic"], "competitor_follow": 1.0, "competitor_noise_sd": 0.001}
    collinear = make_synthetic_panel(master, cfg)
    res = fit_elasticity_model(collinear, REG)
    assert res.used_ridge and res.model.kind == "Ridge"
    assert max(res.vif.values()) > REG["vif_threshold"]


def test_features_use_given_reference_prices(panel):
    train, test = chronological_split(panel, 0.3)
    _, ref = add_features(train)
    feats, ref2 = add_features(test, ref=ref)
    assert ref2.equals(ref)                                   # test never builds its own reference


def test_predict_demand_decreases_with_price(panel):
    res = fit_elasticity_model(panel, REG)
    row = panel.iloc[0]
    kw = dict(competitor_price=row["competitor_price_t"], reputation=row["reputation_score"],
              baseline_demand=row["baseline_demand"], product_id=row["product_id"])
    low = res.predict_demand(price=row["price"] * 0.9, **kw)
    high = res.predict_demand(price=row["price"] * 1.1, **kw)
    assert low > high


def test_promo_effect_runs_and_reports_keys():
    weeks = pd.date_range("2015-01-05", periods=150, freq="W-MON")
    rng = np.random.default_rng(1)
    rows = []
    for fam in ("A", "B"):
        promo = rng.integers(0, 50, len(weeks)).astype(float)
        sales = 1000 * np.exp(0.1 * np.log1p(promo) + rng.normal(0, 0.05, len(weeks)))
        rows.append(pd.DataFrame({"product_id": fam, "week": weeks, "sales": sales,
                                  "onpromotion": promo, "is_outlier": False}))
    out = fit_promo_effect(pd.concat(rows), test_weeks=30)
    assert out["promo_coefficient"] > 0
    assert {"test_R2_log", "test_MAE_log", "n_train", "n_test"} <= set(out)
    assert "NOT a price elasticity" in out["type"]
