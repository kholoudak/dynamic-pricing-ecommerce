"""Tests for src/pipeline/pipeline.py (Agent 3, Phase 6).

The end-to-end test builds small MOCK input files and points the config at them. It proves the
pipeline works with mock data and that real files can replace them without code changes.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.pipeline.pipeline import (
    OUTPUT_COLUMNS, PipelineError, audit_recommendations, load_recommendations, load_run_info,
    run_pipeline,
)
from src.pricing.config import load_config


@pytest.fixture
def mock_cfg(tmp_path):
    rng = np.random.default_rng(3)
    fams = ["A", "B", "C"]
    fc_weeks = pd.date_range("2017-08-14", periods=4, freq="W-MON")
    pd.DataFrame([{"product_id": f, "week": w, "demand_pred": 1000.0 * (i + 1) + 10 * k, "horizon": k + 1}
                  for i, f in enumerate(fams) for k, w in enumerate(fc_weeks)]).to_csv(tmp_path / "forecast.csv", index=False)

    hist_weeks = pd.date_range("2015-01-05", periods=130, freq="W-MON")
    pd.DataFrame([{"product_id": f, "week": w, "sales": 1000.0 * (i + 1) * rng.uniform(0.9, 1.1),
                   "onpromotion": float(rng.integers(0, 20)), "is_outlier": False}
                  for i, f in enumerate(fams) for w in hist_weeks]).to_csv(tmp_path / "sales.csv", index=False)

    pd.DataFrame({"product_name": [f"prod {i}" for i in range(6)], "reputation_score": rng.uniform(0.4, 0.9, 6),
                  "n_reviews": [100] * 6}).to_csv(tmp_path / "reputation.csv", index=False)

    price = rng.uniform(20, 200, 9)
    pd.DataFrame({"product_id": range(1, 10), "current_price": price, "competitor_price": price * 1.03,
                  "product_cost": price * rng.uniform(0.55, 0.85, 9)}).to_csv(tmp_path / "products.csv", index=False)

    cfg = copy.deepcopy(load_config())
    cfg["paths"].update(forecast_file=str(tmp_path / "forecast.csv"), sales_file=str(tmp_path / "sales.csv"),
                        reputation_file=str(tmp_path / "reputation.csv"), products_file=str(tmp_path / "products.csv"),
                        mapping_file=str(tmp_path / "mapping.csv"), master_file=str(tmp_path / "master.csv"),
                        recommendations_file=str(tmp_path / "out" / "recommended_prices.csv"),
                        reports_dir=str(tmp_path / "reports"))
    return cfg


def test_pipeline_end_to_end_on_mock_data(mock_cfg):
    res = run_pipeline(mock_cfg)
    rec = res.recommendations
    assert len(rec) == 9 * 4
    assert list(rec.columns) == OUTPUT_COLUMNS
    assert OUTPUT_COLUMNS[:5] == ["product_id", "week", "price_current", "price_recommended", "margin_gain"]
    assert (rec["price_recommended"] > 0).all() and rec["price_recommended"].notna().all()
    assert audit_recommendations(rec, mock_cfg) == []
    assert "ASSUMED elasticity" in res.metadata["note"]
    assert res.metadata["regression"] is not None


def test_exported_files_can_be_reloaded(mock_cfg):
    run_pipeline(mock_cfg, run_regression=False)
    again = load_recommendations(mock_cfg)
    assert len(again) == 36 and pd.api.types.is_datetime64_any_dtype(again["week"])
    info = load_run_info(mock_cfg)
    assert info["n_products"] == 9 and "generated_at" in info


def test_missing_input_file_gives_a_clear_error(mock_cfg, tmp_path):
    mock_cfg["paths"]["forecast_file"] = str(tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        run_pipeline(mock_cfg, run_regression=False)


def test_load_recommendations_before_running_explains_what_to_do(mock_cfg):
    with pytest.raises(FileNotFoundError, match="Run the pipeline"):
        load_recommendations(mock_cfg)


def test_audit_catches_violations(mock_cfg):
    rec = run_pipeline(mock_cfg, run_regression=False, save=False).recommendations.copy()
    bad = rec.copy()
    bad.loc[0, "price_recommended"] = bad.loc[0, "cost"] * 0.9              # below the minimum margin
    assert any("minimum margin" in p for p in audit_recommendations(bad, mock_cfg))
    bad = rec.copy()
    bad.loc[0, "price_recommended"] = bad.loc[0, "price_current"] * 3       # far above the ceiling
    assert any("ceiling" in p for p in audit_recommendations(bad, mock_cfg))


def test_pipeline_raises_if_audit_fails(mock_cfg, monkeypatch):
    import src.pipeline.pipeline as pl
    monkeypatch.setattr(pl, "recommend_all", _broken)          # swap in a deliberately wrong optimizer
    with pytest.raises(PipelineError):
        run_pipeline(mock_cfg, run_regression=False, save=False)


def _broken(master, cfg):
    """A deliberately wrong optimizer: recommends a price below cost."""
    from src.pricing.optimizer import recommend_all
    rec = recommend_all(master, cfg)
    rec["price_recommended"] = rec["cost"] * 0.5
    return rec
