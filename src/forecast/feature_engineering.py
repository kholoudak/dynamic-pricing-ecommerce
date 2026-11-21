"""
Step 5a - Feature engineering / Etape 5a - Creation des variables.

Input : outputs/sales_weekly.csv (product_id, week, sales, onpromotion, open_stores, is_outlier)
Output: one row per (product, week) with the model features, ALL computed from the current
        week and the PAST only (no feature looks at a future week).

Leakage rules applied here
- lags use shift(k), k >= 1; rolling statistics use the current week and the weeks BEFORE it;
- `promo_available` is causal (cumulative);
- `is_outlier` is NOT a feature: it is computed with a centred window (it sees future weeks);
- holidays are calendar facts, known in advance.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HOLIDAY_TYPES = {"Holiday", "Additional", "Bridge", "Transfer"}  # real national days off / celebrations
EVENT_TYPES = {"Event"}                                          # commercial / special events
FEATURE_DEFAULTS = {"lags": [1, 2, 4], "rolling_windows": [4, 8]}


def get_feature_params(config: dict) -> dict:
    """Return the `features` settings of config.yaml, completed with defaults."""
    return {**FEATURE_DEFAULTS, **(config.get("features") or {})}


def check_weekly_continuity(weekly: pd.DataFrame) -> Dict[str, List[pd.Timestamp]]:
    """Find, per product, the Mondays missing between its first and last week.

    Output: {product_id: [missing weeks]} (empty dict = every series is continuous).
    A gap would silently shift lags, rolling windows and sliding windows.
    """
    gaps = {}
    for pid, g in weekly.groupby("product_id"):
        full = pd.date_range(g["week"].min(), g["week"].max(), freq="7D")
        missing = full.difference(pd.DatetimeIndex(g["week"]))
        if len(missing):
            gaps[pid] = list(missing)
    return gaps


def national_calendar_counts(holidays: pd.DataFrame, weeks: pd.Series) -> pd.DataFrame:
    """Count, for each week (Monday), the national holidays and national events.

    Rules: only locale == National (the series is the national total); rows with
    transferred == True are ignored (the official day was moved, the real day is the
    `Transfer` row); `Work Day` rows (extra working days) are ignored.
    Output columns: week, n_holidays, n_events.
    """
    nat = holidays[(holidays["locale"] == "National") & (~holidays["transferred"].astype(bool))]
    weeks = pd.Series(pd.to_datetime(weeks).unique()).astype("datetime64[ns]").sort_values()

    def count(types: set) -> pd.Series:
        days = pd.to_datetime(nat.loc[nat["type"].isin(types), "date"]).drop_duplicates()
        starts = (days - pd.to_timedelta(days.dt.day_of_week, unit="D")).astype("datetime64[ns]")
        return starts.value_counts()

    hol, ev = count(HOLIDAY_TYPES), count(EVENT_TYPES)
    out = pd.DataFrame({"week": weeks.to_numpy()})
    out["n_holidays"] = out["week"].map(hol).fillna(0).astype(int)
    out["n_events"] = out["week"].map(ev).fillna(0).astype(int)
    return out


def _contains_christmas(week_start: pd.Timestamp) -> int:
    """1 if the 7 days starting on `week_start` contain 25 December, else 0."""
    end = week_start + pd.Timedelta(days=6)
    for year in {week_start.year, end.year}:
        if week_start <= pd.Timestamp(year=year, month=12, day=25) <= end:
            return 1
    return 0


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical week-of-year / month encodings and the Christmas-week flag.

    sin/cos encoding makes week 52 and week 1 neighbours. The date used is the middle of the
    week (Monday + 3 days), so a week belongs to the month where most of its days are.
    """
    out = df.copy()
    mid = out["week"] + pd.Timedelta(days=3)
    woy = mid.dt.isocalendar().week.astype(int)
    out["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    out["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    out["month_sin"] = np.sin(2 * np.pi * mid.dt.month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * mid.dt.month / 12.0)
    xmas = {w: _contains_christmas(w) for w in out["week"].unique()}
    out["is_christmas_week"] = out["week"].map(xmas).astype(int)
    return out


def build_features(weekly: pd.DataFrame, holidays: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, List[str]]:
    """Build the feature table (unscaled) used by the LSTM.

    Features (names are produced from config.yaml -> features):
      log_sales            log(1 + weekly sales): stabilises the scale, and the TARGET
      lag_k                log_sales of k weeks earlier
      roll_mean_w/std_w    mean / std of log_sales over the last w weeks (including this one)
      log_promo            log(1 + weekly onpromotion)
      promo_available      1 once any promotion has been recorded in the whole dataset
                           (0 in 2013: zeros then mean "not recorded", not "no promotion")
      open_stores          mean number of stores that sold something that week
      woy_*, month_*       cyclical calendar
      n_holidays, n_events national holidays / events in the week
      is_christmas_week    the week contains 25 December
    Rows with an undefined lag / rolling value (start of each series) are dropped.
    Output: (feature DataFrame, ordered list of feature column names).
    """
    p = get_feature_params(config)
    df = weekly[["product_id", "week", "sales", "onpromotion", "open_stores"]].copy()
    df["week"] = pd.to_datetime(df["week"]).astype("datetime64[ns]")
    df = df.sort_values(["product_id", "week"]).reset_index(drop=True)

    gaps = check_weekly_continuity(df)
    if gaps:
        raise ValueError(f"Missing weeks inside the series (first products: {list(gaps)[:3]}). "
                         "Re-run Step 4 or fill the gaps before building features.")
    if (df["sales"] < 0).any():
        raise ValueError("Negative sales found: log1p is undefined.")

    df["log_sales"] = np.log1p(df["sales"])
    g = df.groupby("product_id")["log_sales"]
    for k in p["lags"]:
        df[f"lag_{k}"] = g.shift(k)
    for w in p["rolling_windows"]:
        df[f"roll_mean_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        df[f"roll_std_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=w).std())

    df["log_promo"] = np.log1p(df["onpromotion"])
    seen = (df.groupby("week")["onpromotion"].sum().cumsum() > 0).astype(int)  # causal
    df["promo_available"] = df["week"].map(seen).astype(int)

    df = add_calendar_features(df)
    df = df.merge(national_calendar_counts(holidays, df["week"]), on="week", how="left")

    feature_cols = (["log_sales"] + [f"lag_{k}" for k in p["lags"]]
                    + [f"{s}_{w}" for w in p["rolling_windows"] for s in ("roll_mean", "roll_std")]
                    + ["log_promo", "promo_available", "open_stores", "woy_sin", "woy_cos",
                       "month_sin", "month_cos", "n_holidays", "n_events", "is_christmas_week"])
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols
