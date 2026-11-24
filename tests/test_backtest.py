"""Tests for src/pricing/backtest.py (Agent 3, Phase 5)."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.pricing.backtest import (
    LABEL, concentration, make_charts, price_statistics, run_backtest, simulate, total_gain_pct,
)
from src.pricing.config import load_config

CFG = copy.deepcopy(load_config())
CFG["backtest"]["n_runs"] = 5


def small_master():
    rng = np.random.default_rng(0)
    weeks = pd.to_datetime(["2017-08-14", "2017-08-21", "2017-08-28"])
    rows = []
    for i in range(12):
        price = float(rng.uniform(10, 100))
        for w in weeks:
            rows.append(dict(product_id=str(i), category="A" if i % 2 else "B", week=w,
                             price_current=price, cost=price * rng.uniform(0.5, 0.85),
                             demand_pred=float(rng.uniform(100, 5000)), competitor_price=price * 1.02,
                             reputation_score=0.7, reputation_available=True, demand_index=1.0))
    return pd.DataFrame(rows)


def fake_rec(p1):
    return pd.DataFrame({"product_id": ["1"], "category": ["A"], "status": ["ok"], "demand_current": [100.0],
                         "price_current": [10.0], "price_recommended": [p1], "cost": [6.0]})


def test_simulate_formula_and_no_change_means_no_gain():
    s = simulate(fake_rec(10.0), -3.0)
    assert s["profit_static"].iloc[0] == pytest.approx(400.0)        # (10-6)*100
    assert s["gain"].iloc[0] == pytest.approx(0.0)
    up = simulate(fake_rec(11.0), -2.0)                             # (11-6) * 100 * 1.1**-2
    assert up["profit_dynamic"].iloc[0] == pytest.approx(5 * 100 * 1.1 ** -2)


def test_gain_sign_depends_on_true_elasticity():
    # a price rise pays off when demand is not very elastic, and hurts when it is very elastic
    assert total_gain_pct(simulate(fake_rec(11.0), -1.2)) > 0
    assert total_gain_pct(simulate(fake_rec(11.0), -6.0)) < 0


def test_concentration_detects_one_dominant_product():
    sim = pd.DataFrame({"product_id": ["a", "b", "c"], "gain": [100.0, -10.0, -10.0],
                        "profit_static": [1000.0, 1000.0, 1000.0]})
    c = concentration(sim, top_n=1)
    assert c["top_share_of_net_gain_pct"] == pytest.approx(100 / 80 * 100)    # > 100% : others lose
    assert c["gain_excluding_top_pct"] == pytest.approx(-1.0)


def test_price_statistics_static_has_no_variation():
    rec = pd.DataFrame({"product_id": ["1"] * 3, "week": pd.to_datetime(["2017-08-14", "2017-08-21", "2017-08-28"]),
                        "status": "ok", "price_current": 10.0, "price_recommended": [11.0, 11.0, 12.1]})
    st = price_statistics(rec)
    assert st["static_price_variation_pct"] == 0.0
    assert st["mean_abs_weekly_change_pct"] == pytest.approx((10 + 0 + 10) / 3)
    assert st["avg_price_dynamic"] > st["avg_price_static"]


def test_run_backtest_is_labelled_reproducible_and_complete():
    r1 = run_backtest(CFG, master=small_master(), save=False)
    r2 = run_backtest(CFG, master=small_master(), save=False)
    res = r1[0]
    assert res["label"] == LABEL and "SIMULATED" in LABEL
    assert len(res["B_grid"]) == len(CFG["backtest"]["true_elasticity_grid"])
    assert {"A_matched_gain_pct", "C_noisy", "price_statistics"} <= set(res)
    assert res["C_noisy"]["mean_gain_pct"] == pytest.approx(r2[0]["C_noisy"]["mean_gain_pct"])  # same seed
    assert set(r1[3].index) == {"A", "B"}                                                       # by category


def test_charts_are_written(tmp_path):
    res, rec, mc, cat = run_backtest(CFG, master=small_master(), save=False)
    make_charts(rec, res, mc, cat, tmp_path)
    assert len(list(tmp_path.glob("backtest_*.png"))) == 4
