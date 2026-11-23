"""Tests for src/pricing/optimizer.py (Agent 3, Phase 4)."""
import copy

import pandas as pd
import pytest

from src.pricing.config import load_config
from src.pricing.optimizer import (
    expected_demand, expected_profit, optimal_price, product_elasticity, recommend_all,
)

CFG = load_config()


def product(**kw):
    base = dict(product_id="1", price_current=100.0, cost=60.0, demand_pred=1000.0,
                competitor_price=100.0, reputation_score=0.714, reputation_available=True,
                demand_index=1.0)
    base.update(kw)
    return base


def wide_cfg(**opt):
    """Config with very loose constraints and a fixed elasticity, to test the pure maths."""
    c = copy.deepcopy(CFG)
    c["constraints"].update(floor_pct_of_current=0.1, ceiling_pct_of_current=10.0,
                            max_change_per_update_pct=5.0)
    c["optimizer"].update(base_elasticity=-2.0, gap_sensitivity=0, reputation_sensitivity=0,
                          demand_sensitivity=0, **opt)
    return c


def test_grid_matches_analytic_optimum():
    r = optimal_price(product(), wide_cfg())
    assert r["unconstrained_optimum"] == pytest.approx(60 * -2 / (1 + -2))    # = 120
    assert r["price_recommended"] == pytest.approx(120.0, abs=0.1)
    assert r["binding_constraint"] is None


def test_recommended_profit_never_below_current_profit():
    r = optimal_price(product(), CFG)
    assert r["profit_recommended"] >= r["profit_current"] - 1e-6
    assert r["margin_gain"] >= -1e-6


def test_output_has_all_required_fields():
    r = optimal_price(product(), CFG)
    for key in ["price_current", "price_recommended", "demand_current", "demand_recommended",
                "profit_current", "profit_recommended", "margin_gain", "margin_pct_current",
                "margin_pct_recommended", "reasoning", "elasticity_used", "status"]:
        assert key in r
    assert r["status"] == "ok" and len(r["reasoning"]) > 20


def test_demand_falls_when_price_rises():
    assert expected_demand(110, 100, 1000, -2.0) < 1000 < expected_demand(90, 100, 1000, -2.0)
    assert expected_profit(100, 60, 100, 1000, -2.0) == pytest.approx(40 * 1000)


def test_constraints_are_respected():
    r = optimal_price(product(cost=30.0), CFG)               # huge margin room -> wants a big rise
    assert r["price_recommended"] <= 110.0 + 1e-9            # +10% limit
    assert r["binding_constraint"] == "max_change_per_update"


def test_min_margin_respected_even_if_price_is_too_low():
    r = optimal_price(product(cost=98.0), CFG)               # current margin ~2% < required 10%
    assert r["price_recommended"] >= 98.0 * 1.10 - 1e-9


def test_missing_cost_keeps_price():
    r = optimal_price(product(cost=float("nan")), CFG)
    assert r["status"] == "no_recommendation" and r["price_recommended"] == 100.0
    assert "cost" in r["reasoning"]


def test_invalid_price_raises():
    with pytest.raises(ValueError):
        optimal_price(product(price_current=-4), CFG)


def test_elasticity_adjustments_have_the_right_direction():
    opt = CFG["optimizer"]
    base, _ = product_elasticity(100, 100, 0.714, True, 1.0, opt)
    pricier, _ = product_elasticity(100, 90, 0.714, True, 1.0, opt)        # pricier than competitor
    better_rep, _ = product_elasticity(100, 100, 0.9, True, 1.0, opt)
    strong_demand, _ = product_elasticity(100, 100, 0.714, True, 1.3, opt)
    assert abs(pricier) > abs(base) > abs(better_rep)
    assert abs(strong_demand) < abs(base)
    assert -opt["max_abs_elasticity"] <= base <= -opt["min_abs_elasticity"]


def test_unavailable_reputation_is_neutral_and_flagged():
    r = optimal_price(product(reputation_score=0.99, reputation_available=False), CFG)
    assert "reputation_unavailable" in r["flags"]
    neutral = optimal_price(product(reputation_score=CFG["optimizer"]["reputation_reference"]), CFG)
    assert r["elasticity_used"] == pytest.approx(neutral["elasticity_used"])


def test_recommend_all_chains_weeks_within_max_change():
    weeks = pd.to_datetime(["2017-08-14", "2017-08-21", "2017-08-28"])
    master = pd.DataFrame([product(week=w, category="A", cost=30.0) for w in weeks])
    rec = recommend_all(master, CFG)
    prices = [100.0] + rec.sort_values("week")["price_recommended"].tolist()
    for a, b in zip(prices, prices[1:]):
        assert abs(b / a - 1) <= CFG["constraints"]["max_change_per_update_pct"] + 1e-6
    assert prices[-1] <= 120.0 + 1e-9                          # never beyond the ceiling
