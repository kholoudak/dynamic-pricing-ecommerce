"""Tests for src/pipeline/mapping.py and src/pipeline/integration.py (Agent 3, Phase 2)."""
import pandas as pd
import pytest

from src.pipeline.integration import build_master_table
from src.pipeline.loaders import DataValidationError
from src.pipeline.mapping import build_mapping, validate_mapping
from src.pricing.config import load_config

REQUIRED = load_config()["schema"]["master"]


@pytest.fixture
def data():
    products = pd.DataFrame({"product_id": ["1", "2", "3"], "price_current": [10.0, 20.0, 30.0],
                             "cost": [5.0, 10.0, 20.0], "competitor_price": [11.0, 19.0, 31.0]})
    weeks = pd.to_datetime(["2017-08-14", "2017-08-21"])
    forecast = pd.DataFrame({"product_id": ["FOOD"] * 2 + ["TOYS"] * 2, "week": list(weeks) * 2,
                             "demand_pred": [100.0, 120.0, 40.0, 50.0]})
    reputation = pd.DataFrame({"product_name": ["a", "b"], "reputation_score": [0.9, 0.5],
                               "n_reviews": [100, 10], "low_confidence": [False, True]})
    return products, forecast, reputation


def test_mapping_covers_all_products_and_is_reproducible(data):
    products, forecast, reputation = data
    m1 = build_mapping(products, forecast, reputation, seed=1)
    m2 = build_mapping(products, forecast, reputation, seed=1)
    pd.testing.assert_frame_equal(m1, m2)                       # same seed -> same mapping
    assert set(m1["category"]) <= {"FOOD", "TOYS"}
    assert m1["reputation_name"].notna().sum() == 2             # 2 names, 3 products
    validate_mapping(m1, products, forecast)                    # no error


def test_validate_mapping_rejects_unknown_category(data):
    products, forecast, reputation = data
    m = build_mapping(products, forecast, reputation)
    m.loc[0, "category"] = "UNKNOWN"
    with pytest.raises(DataValidationError):
        validate_mapping(m, products, forecast)


def test_master_table_shape_and_demand_split(data):
    products, forecast, reputation = data
    mapping = build_mapping(products, forecast, reputation, seed=1)
    master = build_master_table(forecast, reputation, products, mapping, REQUIRED).data
    assert len(master) == 3 * 2                                  # 3 products x 2 weeks
    assert not master.duplicated(["product_id", "week"]).any()
    # the demand split between the products of a family must add up to the family forecast
    summed = master.groupby(["category", "week"])["demand_pred"].sum().reset_index()
    expected = forecast.rename(columns={"product_id": "category", "demand_pred": "family_total"})
    check = summed.merge(expected, on=["category", "week"], how="inner")
    assert len(check) == len(summed)
    assert (check["demand_pred"] - check["family_total"]).abs().max() < 1e-9


def test_missing_reputation_is_flagged_and_filled(data):
    products, forecast, reputation = data
    mapping = build_mapping(products, forecast, reputation, seed=1)
    res = build_master_table(forecast, reputation, products, mapping, REQUIRED)
    master = res.data
    assert master["reputation_score"].notna().all()
    assert (~master["reputation_available"]).any()               # one product has no reputation
    assert any("no reputation" in w for w in res.warnings)


def test_missing_required_column_raises(data):
    products, forecast, reputation = data
    mapping = build_mapping(products, forecast, reputation)
    with pytest.raises(DataValidationError):
        build_master_table(forecast, reputation, products, mapping, REQUIRED + ["nonexistent"])
