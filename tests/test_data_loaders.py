"""Tests for src/pipeline/loaders.py (Agent 3, Phase 1).

Each test writes a tiny CSV in a temporary folder, so the real data is never touched.
"""
import pandas as pd
import pytest

from src.pipeline.loaders import (
    DataValidationError,
    load_forecast,
    load_products,
    load_reputation,
    load_sales_history,
)
from src.pricing.config import load_config

CFG = load_config()


def write(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_forecast_ok(tmp_path):
    df = pd.DataFrame({"product_id": ["A", "A"], "week": ["2017-08-14", "2017-08-21"],
                       "demand_pred": [10.0, 12.0]})
    res = load_forecast(CFG, write(tmp_path, "f.csv", df))
    assert len(res.data) == 2 and res.warnings == []
    assert pd.api.types.is_datetime64_any_dtype(res.data["week"])


def test_forecast_missing_column_raises(tmp_path):
    df = pd.DataFrame({"product_id": ["A"], "week": ["2017-08-14"]})
    with pytest.raises(DataValidationError):
        load_forecast(CFG, write(tmp_path, "f.csv", df))


def test_forecast_duplicates_and_negatives(tmp_path):
    df = pd.DataFrame({"product_id": ["A", "A", "B"],
                       "week": ["2017-08-14", "2017-08-14", "2017-08-14"],
                       "demand_pred": [5.0, 6.0, -3.0]})
    res = load_forecast(CFG, write(tmp_path, "f.csv", df))
    assert len(res.data) == 2                      # duplicate removed
    assert res.data["demand_pred"].min() == 0      # negative clipped
    assert len(res.warnings) == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_forecast(CFG, tmp_path / "nope.csv")


def test_reputation_without_product_id_warns(tmp_path):
    df = pd.DataFrame({"product_name": ["x", "y"], "reputation_score": [0.9, 1.4],
                       "n_reviews": [100, 5]})
    res = load_reputation(CFG, write(tmp_path, "r.csv", df))
    assert any("product_id" in w for w in res.warnings)
    assert res.data["reputation_score"].max() == 1.0          # clipped
    assert res.data["low_confidence"].tolist() == [False, True]


def test_products_renamed_and_invalid_cost(tmp_path):
    df = pd.DataFrame({"product_id": [1, 2, 3], "current_price": [10.0, 20.0, 0.0],
                       "competitor_price": [11.0, 19.0, 5.0], "product_cost": [-1.0, 12.0, 3.0]})
    res = load_products(CFG, write(tmp_path, "p.csv", df))
    assert {"price_current", "cost"} <= set(res.data.columns)
    assert len(res.data) == 2                                  # price 0 dropped
    assert pd.isna(res.data.loc[res.data["product_id"] == "1", "cost"]).all()


def test_products_duplicate_ids(tmp_path):
    df = pd.DataFrame({"product_id": [1, 1], "current_price": [10.0, 11.0],
                       "competitor_price": [9.0, 9.0], "product_cost": [5.0, 5.0]})
    res = load_products(CFG, write(tmp_path, "p.csv", df))
    assert len(res.data) == 1


def test_sales_zero_weeks_flagged(tmp_path):
    df = pd.DataFrame({"product_id": ["A", "A"], "week": ["2017-01-02", "2017-01-09"],
                       "sales": [0.0, 5.0]})
    res = load_sales_history(CFG, write(tmp_path, "s.csv", df))
    assert any("zero sales" in w for w in res.warnings)
