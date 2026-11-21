"""Unit tests for feature_engineering.py and sequence_generator.py (run: pytest tests -v)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecast import data_loader as dl  # noqa: E402
from src.forecast import feature_engineering as fe  # noqa: E402
from src.forecast import sequence_generator as sg  # noqa: E402

N_WEEKS = 120


@pytest.fixture
def config():
    return dl.load_config()


def make_weekly(n_weeks=N_WEEKS, products=("A", "B", "C"), start="2014-01-06", seed=0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start, periods=n_weeks, freq="7D")
    rows = []
    for k, p in enumerate(products):
        base = 100 * (k + 1) * 10 ** k  # very different scales per product
        for i, w in enumerate(weeks):
            sales = base * (1 + 0.3 * np.sin(2 * np.pi * i / 52)) + rng.normal(0, base * 0.02)
            rows.append((p, w, max(sales, 0), 0 if i < 10 else int(rng.integers(1, 30)), 50 + (i % 3), False))
    return pd.DataFrame(rows, columns=["product_id", "week", "sales", "onpromotion", "open_stores", "is_outlier"])


def make_holidays():
    return pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01", "2014-02-12", "2014-03-05", "2014-03-06", "2014-04-01", "2014-05-05"]),
        "type": ["Holiday", "Holiday", "Holiday", "Event", "Holiday", "Holiday"],
        "locale": ["National", "National", "National", "National", "Local", "National"],
        "locale_name": ["Ecuador"] * 4 + ["Quito", "Ecuador"],
        "description": list("abcdef"),
        "transferred": [False, True, False, False, False, False],
    })


@pytest.fixture
def prepared(config):
    cfg = {**config, "forecast": {"window_size": 12, "horizon": 4}, "split": {"val_weeks": 26, "test_weeks": 26}}
    return cfg, sg.prepare_dataset(cfg, weekly=make_weekly(), holidays=make_holidays(), save=False)


# ------------------------------------------------------------------ feature engineering
def test_continuity_detects_gap():
    w = make_weekly(n_weeks=10)
    w = w.drop(w[(w["product_id"] == "B") & (w["week"] == w["week"].iloc[4])].index)
    gaps = fe.check_weekly_continuity(w)
    assert list(gaps) == ["B"] and len(gaps["B"]) == 1


def test_build_features_raises_on_gap(config):
    w = make_weekly(n_weeks=30)
    w = w.drop(w.index[5])
    with pytest.raises(ValueError):
        fe.build_features(w, make_holidays(), config)


def test_lag_and_rolling_use_only_past(config):
    w = make_weekly(n_weeks=40, products=("A",))
    f1, _ = fe.build_features(w, make_holidays(), config)
    f1 = f1.set_index("week")
    # lag_1 of week t equals log_sales of week t-1
    prev = f1["log_sales"].shift(1)
    assert np.allclose(f1["lag_1"].iloc[1:], prev.iloc[1:])
    # rolling mean of 4 = mean of the current week and the 3 previous ones
    t = f1.index[10]
    expected = f1["log_sales"].iloc[7:11].mean()
    assert f1.loc[t, "roll_mean_4"] == pytest.approx(expected)
    # changing the LAST week must not change any earlier feature
    w2 = w.copy()
    w2.loc[w2.index[-1], "sales"] = 1e9
    f2, cols = fe.build_features(w2, make_holidays(), config)
    f2 = f2.set_index("week")
    assert np.allclose(f1.iloc[:-1][cols].to_numpy(float), f2.iloc[:-1][cols].to_numpy(float))


def test_start_rows_with_undefined_features_are_dropped(config):
    w = make_weekly(n_weeks=40, products=("A",))
    f, _ = fe.build_features(w, make_holidays(), config)
    assert len(f) == 40 - 7  # largest window = 8 weeks -> 7 weeks lost
    assert not f.isna().any().any()


def test_holiday_counts_national_only_and_ignore_transferred():
    weeks = pd.Series(pd.date_range("2013-12-30", periods=14, freq="7D"))  # Mondays
    out = fe.national_calendar_counts(make_holidays(), weeks).set_index("week")
    assert out.loc["2013-12-30", "n_holidays"] == 1       # 2014-01-01 national holiday
    assert out.loc["2014-02-10", "n_holidays"] == 0       # 2014-02-12 is transferred=True
    assert out.loc["2014-03-03", "n_holidays"] == 1 and out.loc["2014-03-03", "n_events"] == 1
    assert out.loc["2014-03-31", "n_holidays"] == 0       # 2014-04-01 is local only


def test_christmas_week_flag():
    assert fe._contains_christmas(pd.Timestamp("2014-12-22")) == 1  # 22-28 Dec
    assert fe._contains_christmas(pd.Timestamp("2014-12-29")) == 0
    assert fe._contains_christmas(pd.Timestamp("2014-12-15")) == 0


def test_promo_available_is_causal(config):
    w = make_weekly(n_weeks=40, products=("A",))
    f, _ = fe.build_features(w, make_holidays(), config)
    first_on = w.loc[w["onpromotion"] > 0, "week"].min()
    assert (f.loc[f["week"] < first_on, "promo_available"] == 0).all()
    assert (f.loc[f["week"] >= first_on, "promo_available"] == 1).all()


def test_outlier_flag_is_not_a_feature(prepared):
    _, res = prepared
    assert "is_outlier" not in res["feature_cols"]


# ------------------------------------------------------------------ split / scaler / windows
def test_split_dates_are_chronological():
    weeks = pd.Series(pd.date_range("2014-01-06", periods=100, freq="7D"))
    d = sg.compute_split_dates(weeks, 26, 26)
    assert d["train_end"] < d["val_end"] < d["test_end"]
    assert (weeks > d["val_end"]).sum() == 26 and ((weeks > d["train_end"]) & (weeks <= d["val_end"])).sum() == 26


def test_split_too_short_raises():
    with pytest.raises(ValueError):
        sg.compute_split_dates(pd.Series(pd.date_range("2014-01-06", periods=40, freq="7D")), 26, 26)


def test_no_target_leakage_between_splits(prepared):
    _, res = prepared
    d, seq = res["split_dates"], res["sequences"]
    assert (seq["train"]["meta"]["last_target_week"] <= d["train_end"]).all()
    assert (seq["val"]["meta"]["first_target_week"] > d["train_end"]).all()
    assert (seq["val"]["meta"]["last_target_week"] <= d["val_end"]).all()
    assert (seq["test"]["meta"]["first_target_week"] > d["val_end"]).all()
    assert all(len(seq[s]["X"]) > 0 for s in sg.SPLITS)


def test_sequence_shapes_and_alignment(prepared):
    cfg, res = prepared
    seq, cols = res["sequences"]["train"], res["feature_cols"]
    assert seq["X"].shape[1:] == (12, len(cols)) and seq["y"].shape[1] == 4
    scaled = res["scaled"]
    row = seq["meta"].iloc[5]
    g = scaled[scaled["product_id"] == row["product_id"]].set_index("week")
    origin = pd.Timestamp(row["origin_week"])
    # last input step = origin week; target = the 4 following weeks
    assert np.allclose(seq["X"][5][-1], g.loc[origin, cols].to_numpy(np.float32))
    nxt = g.loc[origin + pd.Timedelta(days=7): origin + pd.Timedelta(days=28), "log_sales"].to_numpy(np.float32)
    assert np.allclose(seq["y"][5], nxt)


def test_scaler_fitted_on_train_only(config):
    cfg = {**config, "forecast": {"window_size": 12, "horizon": 4}, "split": {"val_weeks": 26, "test_weeks": 26}}
    base = make_weekly()
    res1 = sg.prepare_dataset(cfg, weekly=base, holidays=make_holidays(), save=False)
    changed = base.copy()
    last_weeks = sorted(changed["week"].unique())[-52:]  # val + test period
    changed.loc[changed["week"].isin(last_weeks), "sales"] *= 50
    res2 = sg.prepare_dataset(cfg, weekly=changed, holidays=make_holidays(), save=False)
    assert res1["scaler"].params == res2["scaler"].params
    assert res1["scaler"].open_stores == res2["scaler"].open_stores


def test_scaled_train_has_zero_mean_per_product(prepared):
    _, res = prepared
    scaled, d = res["scaled"], res["split_dates"]
    tr = scaled[scaled["week"] <= d["train_end"]]
    means = tr.groupby("product_id")["log_sales"].mean()
    assert np.allclose(means, 0, atol=1e-6)


def test_inverse_sales_roundtrip(prepared):
    _, res = prepared
    feats, scaled, scaler = res["features"], res["scaled"], res["scaler"]
    back = scaler.inverse_sales(scaled["product_id"], scaled["log_sales"])
    assert np.allclose(back, feats["sales"], rtol=1e-4)
    two_d = scaler.inverse_sales(scaled["product_id"].iloc[:6], np.tile(scaled["log_sales"].iloc[:6].to_numpy()[:, None], (1, 4)))
    assert two_d.shape == (6, 4) and (two_d >= 0).all()


def test_scaler_unknown_product_raises(prepared):
    _, res = prepared
    df = res["features"].copy()
    df["product_id"] = "ZZZ"
    with pytest.raises(ValueError):
        res["scaler"].transform(df)


def test_scaler_save_load(prepared, tmp_path):
    _, res = prepared
    res["scaler"].save(tmp_path / "s.json")
    loaded = sg.FeatureScaler.load(tmp_path / "s.json")
    assert loaded.params == res["scaler"].params and loaded.feature_cols == res["scaler"].feature_cols


def test_save_and_reload_roundtrip(config, tmp_path):
    cfg = {**config, "forecast": {"window_size": 12, "horizon": 4}, "split": {"val_weeks": 26, "test_weeks": 26}}
    res = sg.prepare_dataset(cfg, weekly=make_weekly(), holidays=make_holidays(), out_root=tmp_path, save=True)
    loaded = sg.load_prepared(cfg, root=tmp_path)
    for s in sg.SPLITS:
        assert np.array_equal(loaded["sequences"][s]["X"], res["sequences"][s]["X"])
        assert len(loaded["sequences"][s]["meta"]) == len(res["sequences"][s]["X"])
    assert loaded["info"]["products"] == ["A", "B", "C"]
