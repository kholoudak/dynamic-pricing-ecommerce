"""Unit tests for src/forecast/data_loader.py (run: pytest tests -v)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecast import data_loader as dl  # noqa: E402


@pytest.fixture
def config():
    return dl.load_config()


def make_train(dates, stores=(1, 2), families=("A", "B")):
    rows = [(d, s, f, 1.0, 0) for d in dates for s in stores for f in families]
    return pd.DataFrame(rows, columns=["date", "store_nbr", "family", "sales", "onpromotion"])


def test_load_config_has_expected_sections(config):
    for key in ("paths", "data", "forecast", "split", "model"):
        assert key in config


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        dl.load_config(Path("does_not_exist.yaml"))


def test_validate_columns_detects_missing():
    with pytest.raises(ValueError):
        dl.validate_columns(pd.DataFrame({"a": [1]}), ["a", "b"], "demo")


def test_find_missing_dates_detects_gap():
    dates = pd.Series(pd.to_datetime(["2016-12-23", "2016-12-24", "2016-12-26"]))
    assert list(dl.find_missing_dates(dates)) == [pd.Timestamp("2016-12-25")]


def test_find_missing_dates_no_gap():
    dates = pd.Series(pd.date_range("2016-01-01", periods=5))
    assert len(dl.find_missing_dates(dates)) == 0


def test_panel_complete(config):
    df = make_train(pd.date_range("2016-01-01", periods=3))
    r = dl.check_panel_completeness(df, config)
    assert r["is_complete"] and r["expected_rows"] == 3 * 2 * 2 and r["duplicated_keys"] == 0


def test_panel_incomplete_and_duplicates(config):
    df = make_train(pd.date_range("2016-01-01", periods=3))
    df = pd.concat([df.iloc[1:], df.iloc[[1]]])  # drop one row, duplicate another
    r = dl.check_panel_completeness(df, config)
    assert not r["is_complete"] and r["duplicated_keys"] == 1
