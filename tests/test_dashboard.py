"""Tests for src/pipeline/dashboard.py (Agent 3, Phase 8).

Part 1 tests the pure data functions. Part 2 runs the real app headless (Streamlit's AppTest).
"""
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest                       # noqa: E402

from src.pipeline.dashboard import category_summary, compute_kpis, filter_data, fmt_big   # noqa: E402
from src.pricing.config import load_config, resolve_path        # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "pipeline" / "dashboard.py"


def sample():
    w1, w2 = pd.Timestamp("2017-08-14"), pd.Timestamp("2017-08-21")
    return pd.DataFrame({
        "product_id": ["1", "2", "3", "1"], "category": ["A", "A", "B", "A"], "week": [w1, w1, w1, w2],
        "status": ["ok", "ok", "ok", "ok"], "price_current": [10.0, 20.0, 30.0, 10.0],
        "price_recommended": [11.0, 18.0, 30.0, 11.0], "price_change_pct": [10.0, -10.0, 0.0, 10.0],
        "demand_current": [100.0, 50.0, 10.0, 100.0], "demand_recommended": [80.0, 60.0, 10.0, 80.0],
        "profit_current": [400.0, 400.0, 100.0, 400.0], "profit_recommended": [440.0, 420.0, 100.0, 440.0],
        "margin_gain": [40.0, 20.0, 0.0, 40.0], "margin_pct_current": [0.4, 0.3, 0.2, 0.4],
        "margin_pct_recommended": [0.45, 0.25, 0.2, 0.45], "reputation_score": [0.8, 0.5, 0.714, 0.8],
        "reputation_available": [True, True, False, True],
        "binding_constraint": ["max_change_per_update", None, None, "max_change_per_update"],
    })


def test_filter_by_category_and_week():
    df = sample()
    assert len(filter_data(df, ["A"], "2017-08-14")) == 2
    assert len(filter_data(df, None, "2017-08-21")) == 1
    assert len(filter_data(df, ["B"], None)) == 1
    df.loc[0, "status"] = "no_recommendation"
    assert len(filter_data(df, ["A"], "2017-08-14")) == 1          # rows without a recommendation are hidden


def test_kpis():
    k = compute_kpis(filter_data(sample(), None, "2017-08-14"))
    assert k["n_products"] == 3 and k["n_raised"] == 1 and k["n_lowered"] == 1 and k["n_limited"] == 1
    assert k["gain"] == pytest.approx(60.0) and k["gain_pct"] == pytest.approx(60 / 900 * 100)
    assert k["avg_reputation"] == pytest.approx(0.65)               # product 3 (not available) is excluded
    assert k["n_without_reputation"] == 1
    assert compute_kpis(sample().iloc[0:0]) is None


def test_category_summary():
    cs = category_summary(filter_data(sample(), None, "2017-08-14")).set_index("category")
    assert cs.loc["A", "products"] == 2 and cs.loc["A", "gain"] == pytest.approx(60.0)
    assert cs.loc["A", "gain_pct"] == pytest.approx(60 / 800 * 100)


def test_fmt_big():
    assert fmt_big(1_234_567) == "1.2 M" and fmt_big(950) == "950.0" and fmt_big(-2500) == "-2.5 K"
    assert fmt_big(None) == "n/a"


# ---- headless run of the real app (needs the exported recommendations from the pipeline) ----

needs_data = pytest.mark.skipif(not resolve_path(load_config()["paths"]["recommendations_file"]).exists(),
                                reason="run `python -m src.pipeline.pipeline` first")


@needs_data
def test_app_renders_without_errors():
    at = AppTest.from_file(str(DASHBOARD), default_timeout=90).run()
    assert not at.exception
    assert len(at.metric) >= 7                                       # KPI cards
    assert any("SIMULATED" in w.value for w in at.warning)           # honesty banner is always visible
    assert len(at.tabs) == 4


@needs_data
def test_app_category_filter_and_product_selection():
    at = AppTest.from_file(str(DASHBOARD), default_timeout=90).run()
    category = at.sidebar.multiselect[0].options[0]
    at.sidebar.multiselect[0].select(category).run()
    assert not at.exception
    products = [s for s in at.selectbox if s.label == "Product"][0]
    assert len(products.options) >= 1
    products.select_index(len(products.options) - 1).run()
    assert not at.exception
    assert any("Why did the price change" in m.value for m in at.markdown)
