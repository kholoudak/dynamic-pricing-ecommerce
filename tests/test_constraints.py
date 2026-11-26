"""Tests for src/pricing/constraints.py (Agent 3, Phase 4)."""
import pytest

from src.pricing.constraints import PriceConstraints, apply_constraints, price_bounds, resolve_cost

R = PriceConstraints()          # defaults: margin 10%, floor 80%, ceiling 120%, change +/-10%


def test_bounds_normal_case():
    b = price_bounds(100, 70, R)
    assert b["lo"] == pytest.approx(90) and b["hi"] == pytest.approx(110)
    assert b["lo_source"] == b["hi_source"] == "max_change_per_update"


def test_ceiling_and_floor_beat_the_change_limit_when_last_price_is_far():
    # previous price 118 -> +10% would be 129.8, but the ceiling is 120
    assert price_bounds(100, 70, R, last_price=118)["hi"] == pytest.approx(120)
    assert price_bounds(100, 70, R, last_price=118)["hi_source"] == "price_ceiling"
    assert price_bounds(100, 70, R, last_price=82)["lo"] == pytest.approx(80)


def test_min_margin_overrides_change_limit():
    # cost 95 -> min price 104.5 is ABOVE the lower change limit 90 -> margin sets the floor
    b = price_bounds(100, 95, R)
    assert b["lo"] == pytest.approx(104.5) and b["lo_source"] == "min_margin"


def test_min_margin_wins_over_ceiling():
    b = price_bounds(100, 120, R)                   # needs >= 132 but ceiling is 120
    assert b["lo"] == b["hi"] == pytest.approx(132)
    assert any("overrides_price_ceiling" in f for f in b["flags"])


def test_apply_constraints_clamps_and_rounds():
    assert apply_constraints(150, 100, 70, R)[0] == 110.0
    assert apply_constraints(10, 100, 70, R)[0] == 90.0
    assert apply_constraints(-5, 100, 70, R)[0] > 0            # never negative
    price, flags = apply_constraints(103.456, 100, 70, R)
    assert price == 103.46 and flags == []


def test_missing_cost_keep_current():
    assert resolve_cost(100, float("nan"), R) == (None, "cost_missing_price_kept")
    assert resolve_cost(100, -3, R)[0] is None and resolve_cost(100, None, R)[0] is None


def test_missing_cost_default_ratio():
    rules = PriceConstraints(on_missing_cost="use_default_ratio", default_cost_ratio=0.7)
    cost, flag = resolve_cost(100, None, rules)
    assert cost == pytest.approx(70) and "estimated" in flag


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        PriceConstraints(min_price=0)
    with pytest.raises(ValueError):
        PriceConstraints(floor_pct_of_current=1.3, ceiling_pct_of_current=1.1)
    with pytest.raises(ValueError):
        PriceConstraints(on_missing_cost="use_default_ratio")   # ratio missing
    with pytest.raises(ValueError):
        price_bounds(0, 10, R)
