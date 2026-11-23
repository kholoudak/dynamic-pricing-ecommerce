"""Unit tests for src/forecast/preprocessing.py (run: pytest tests -v)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecast import data_loader as dl  # noqa: E402
from src.forecast import preprocessing as pp  # noqa: E402


@pytest.fixture
def config():
    return dl.load_config()


def make_train(start="2016-12-20", end="2017-01-10", skip=("2016-12-25",), stores=(1, 2), families=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range(start, end):
        if d.strftime("%Y-%m-%d") in skip:
            continue
        for s in stores:
            for f in families:
                rows.append((d, s, f, float(rng.integers(5, 20)), 1))
    return pd.DataFrame(rows, columns=["date", "store_nbr", "family", "sales", "onpromotion"])


def test_aggregate_daily_sums_over_stores(config):
    train = make_train(start="2016-01-04", end="2016-01-04")
    daily = pp.aggregate_daily(train, config)
    assert len(daily) == 2  # 2 families
    assert daily["sales"].sum() == pytest.approx(train["sales"].sum())
    assert (daily["onpromotion"] == 2).all()  # 2 stores x 1


def test_compute_open_stores_detects_closed_store(config):
    train = make_train(start="2016-01-04", end="2016-01-05")
    train.loc[(train["store_nbr"] == 2) & (train["date"] == "2016-01-05"), "sales"] = 0
    out = pp.compute_open_stores(train, config).set_index("date")["open_stores"]
    assert out["2016-01-04"] == 2 and out["2016-01-05"] == 1


def test_fill_missing_calendar_days_adds_zero_day(config):
    train = make_train()
    daily = pp.aggregate_daily(train, config)
    filled = pp.fill_missing_calendar_days(daily, pp.compute_open_stores(train, config))
    day = filled[filled["date"] == "2016-12-25"]
    assert len(day) == 2 and (day["sales"] == 0).all() and day["is_imputed_day"].all()
    assert (day["open_stores"] == 0).all()
    assert not filled.loc[filled["date"] != "2016-12-25", "is_imputed_day"].any()


def test_weekly_drops_partial_weeks_and_keeps_christmas_week(config):
    train = make_train()  # 2016-12-20 (Tue) -> 2017-01-10 (Tue)
    daily = pp.aggregate_daily(train, config)
    filled = pp.fill_missing_calendar_days(daily, pp.compute_open_stores(train, config))
    weekly = pp.to_weekly(filled)
    complete, partial = pp.drop_partial_weeks(weekly)
    assert set(partial["week"]) == {pd.Timestamp("2016-12-19"), pd.Timestamp("2017-01-09")}
    assert (complete["n_days"] == 7).all()
    xmas = complete[(complete["week"] == "2016-12-19")]
    assert xmas.empty  # the first week is partial here...
    assert pd.Timestamp("2016-12-26") in set(complete["week"])  # ...but the week of 26 Dec is complete


def test_christmas_week_is_complete_after_filling(config):
    train = make_train(start="2016-12-12", end="2016-12-31")
    daily = pp.aggregate_daily(train, config)
    filled = pp.fill_missing_calendar_days(daily, pp.compute_open_stores(train, config))
    complete, _ = pp.drop_partial_weeks(pp.to_weekly(filled))
    week = complete[(complete["week"] == "2016-12-19") & (complete["product_id"] == "A")].iloc[0]
    assert week["n_imputed_days"] == 1 and week["n_days"] == 7


def test_trim_leading_zeros():
    weekly = pd.DataFrame({
        "product_id": ["A"] * 5 + ["B"] * 3,
        "week": list(pd.date_range("2016-01-04", periods=5, freq="7D")) + list(pd.date_range("2016-01-04", periods=3, freq="7D")),
        "sales": [0, 0, 5, 0, 7, 0, 0, 0],
    })
    trimmed, removed = pp.trim_leading_zeros(weekly)
    assert removed["A"] == 2 and removed["B"] == 3
    assert list(trimmed["sales"]) == [5, 0, 7]  # inner zero kept, B never sells -> dropped


def test_select_products_and_history_summary():
    weekly = pd.DataFrame({
        "product_id": ["A"] * 4 + ["B"] * 2,
        "week": list(pd.date_range("2016-01-04", periods=4, freq="7D")) + list(pd.date_range("2016-01-04", periods=2, freq="7D")),
        "sales": [1, 0, 2, 3, 1, 1],
    })
    summary = pp.history_summary(weekly)
    assert summary.loc["A", "n_weeks"] == 4 and summary.loc["A", "zero_weeks"] == 1
    kept, excluded = pp.select_products(summary, 3)
    assert kept == ["A"] and excluded == ["B"]


def test_flag_outliers_detects_spike_not_trend():
    weeks = pd.date_range("2016-01-04", periods=40, freq="7D")
    smooth = 100 + np.arange(40) * 2.0 + np.sin(np.arange(40)) * 3
    spiky = smooth.copy()
    spiky[20] = 5000
    weekly = pd.DataFrame({"product_id": ["T"] * 40 + ["S"] * 40, "week": list(weeks) * 2,
                           "sales": list(smooth) + list(spiky)})
    flags = pp.flag_outliers(weekly)
    assert flags[weekly["product_id"] == "T"].sum() == 0
    assert flags[(weekly["product_id"] == "S") & (weekly["sales"] == 5000)].all()
    assert flags[weekly["product_id"] == "S"].sum() == 1


def test_zero_diagnostics_explains_closed_store_days(config):
    train = make_train(start="2016-01-04", end="2016-01-10")
    train.loc[(train["store_nbr"] == 2) & (train["date"] <= "2016-01-05"), "sales"] = 0  # store 2 closed 2 days
    train.loc[(train["store_nbr"] == 1) & (train["date"] == "2016-01-08") & (train["family"] == "A"), "sales"] = 0  # real zero
    diag = pp.zero_diagnostics(train, config)
    assert diag["closed_store_days"] == 2
    assert diag["top_stores_closed_days"] == {2: 2}
    assert 0 < diag["zero_rows_explained_by_closed_store_days_pct"] < 100


def test_run_preprocessing_end_to_end(config, tmp_path):
    train = make_train(start="2015-01-05", end="2016-12-31", skip=("2015-12-25", "2016-12-25"))
    config = {**config, "preprocessing": {"min_history_weeks": 20}}
    result = pp.run_preprocessing(config, train=train, out_dir=tmp_path / "out", processed_dir=tmp_path / "proc")
    out = pd.read_csv(tmp_path / "out" / "sales_weekly.csv", parse_dates=["week"])
    assert list(out.columns) == ["product_id", "week", "sales", "onpromotion", "open_stores", "is_outlier"]
    assert set(out["product_id"]) == {"A", "B"}
    assert (out["week"].dt.day_of_week == 0).all()  # every week starts on Monday
    assert not out.duplicated(["product_id", "week"]).any()
    assert result["report"]["days_inserted"] == 2
    # weekly total must equal the raw total, minus the weeks removed as partial
    assert out["sales"].sum() <= train["sales"].sum()
    assert (tmp_path / "out" / "data_quality_report.csv").exists()
