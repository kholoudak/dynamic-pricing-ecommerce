"""Configurable pricing constraints (Agent 3, Phase 4).

All rules come from the ``constraints`` block of pricing_config.yaml.

Two kinds of rules:
  HARD rules (never broken):  minimum price (> 0), minimum margin, price floor, price ceiling.
  SOFT rule  (can yield):     maximum change per update versus the previous price.

If the soft rule collides with a hard rule (for example the current price is already below the
minimum margin, so even a +10% step cannot repair it) the hard rule wins and a flag says so.
If two hard rules collide (minimum margin above the ceiling), the minimum margin wins:
we never recommend a price that loses the required margin.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PriceConstraints:
    """The set of pricing rules."""
    min_margin_pct: float = 0.10
    floor_pct_of_current: float = 0.80
    ceiling_pct_of_current: float = 1.20
    max_change_per_update_pct: float = 0.10
    min_price: float = 0.01
    on_missing_cost: str = "keep_current"
    default_cost_ratio: float = None

    def __post_init__(self):
        if self.min_price <= 0:
            raise ValueError("min_price must be > 0 (prices can never be zero or negative).")
        if not 0 <= self.min_margin_pct < 1:
            raise ValueError("min_margin_pct must be in [0, 1).")
        if self.floor_pct_of_current <= 0 or self.ceiling_pct_of_current < self.floor_pct_of_current:
            raise ValueError("Need 0 < floor_pct_of_current <= ceiling_pct_of_current.")
        if self.max_change_per_update_pct < 0:
            raise ValueError("max_change_per_update_pct must be >= 0.")
        if self.on_missing_cost not in ("keep_current", "use_default_ratio"):
            raise ValueError("on_missing_cost must be 'keep_current' or 'use_default_ratio'.")
        if self.on_missing_cost == "use_default_ratio" and not self.default_cost_ratio:
            raise ValueError("use_default_ratio needs a default_cost_ratio in the config.")

    @classmethod
    def from_config(cls, cfg):
        """Build the rules from the loaded pricing_config.yaml dictionary."""
        return cls(**cfg["constraints"])


def resolve_cost(price_current, cost, rules):
    """Return (usable_cost or None, flag or None).

    A missing, NaN, zero or negative cost is invalid. Depending on ``on_missing_cost`` we either
    give up (None -> the price will not be changed) or estimate cost = price * default ratio.
    """
    if cost is not None and not (isinstance(cost, float) and math.isnan(cost)) and cost > 0:
        return float(cost), None
    if rules.on_missing_cost == "use_default_ratio":
        return float(price_current) * rules.default_cost_ratio, "cost_missing_estimated_from_default_ratio"
    return None, "cost_missing_price_kept"


def price_bounds(price_current, cost, rules, last_price=None):
    """Compute the allowed price interval.

    Returns a dict: lo, hi, lo_source, hi_source (the rule that sets each end), flags.
    ``last_price`` is the previous price in a sequence of updates (defaults to the current price).
    """
    p0 = float(price_current)
    if not p0 > 0:
        raise ValueError(f"price_current must be positive, got {price_current}")
    ref = float(last_price) if last_price else p0
    flags = []

    # hard rules
    hard_lo = {"min_price": rules.min_price, "price_floor": rules.floor_pct_of_current * p0}
    if cost is not None:
        hard_lo["min_margin"] = cost * (1 + rules.min_margin_pct)
    lo_src = max(hard_lo, key=hard_lo.get)
    h_lo = hard_lo[lo_src]
    h_hi, hi_src = rules.ceiling_pct_of_current * p0, "price_ceiling"
    if h_lo > h_hi:                                   # two hard rules collide: margin wins
        flags.append(f"{lo_src}_overrides_price_ceiling")
        h_hi, hi_src = h_lo, lo_src

    # soft rule: maximum change versus the previous price, clamped inside the hard interval
    s_lo = ref * (1 - rules.max_change_per_update_pct)
    s_hi = ref * (1 + rules.max_change_per_update_pct)
    lo, lo_source = (s_lo, "max_change_per_update") if h_lo <= s_lo <= h_hi else \
        ((h_lo, lo_src) if s_lo < h_lo else (h_hi, hi_src))
    hi, hi_source = (s_hi, "max_change_per_update") if h_lo <= s_hi <= h_hi else \
        ((h_lo, lo_src) if s_hi < h_lo else (h_hi, hi_src))
    if lo_source != "max_change_per_update" and s_lo > h_hi:
        flags.append("max_change_limit_relaxed")
    if hi_source != "max_change_per_update" and s_hi < h_lo:
        flags.append("max_change_limit_relaxed")

    return {"lo": lo, "hi": hi, "lo_source": lo_source, "hi_source": hi_source, "flags": flags}


def _round_inside(price, lo, hi):
    """Round to cents without leaving [lo, hi] (if the interval holds no whole cent, keep 4 decimals)."""
    lo_c, hi_c = math.ceil(lo * 100 - 1e-9) / 100, math.floor(hi * 100 + 1e-9) / 100
    if lo_c <= hi_c:
        return float(min(max(round(price, 2), lo_c), hi_c))
    return round(float(min(max(price, lo), hi)), 4)


def apply_constraints(price, price_current, cost, rules, last_price=None):
    """Force any proposed price into the allowed interval. Returns (final_price, flags)."""
    if price is None or not np.isfinite(price):
        raise ValueError("price must be a finite number.")
    cost, cost_flag = resolve_cost(price_current, cost, rules)
    b = price_bounds(price_current, cost, rules, last_price)
    flags = list(b["flags"]) + ([cost_flag] if cost_flag else [])
    final = _round_inside(price, b["lo"], b["hi"])
    if abs(final - price) > 0.005:
        flags.append("price_adjusted_to_respect_constraints")
    return final, flags
