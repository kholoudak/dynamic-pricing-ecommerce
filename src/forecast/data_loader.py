"""
Step 3 - Data loading / Etape 3 - Chargement des donnees.

All paths and column names come from config.yaml (nothing hardcoded here).
Column names used for train.csv were CONFIRMED by the Step 1 inspection.
oil.csv and transactions.csv were NOT inspected yet: only the `date` column
is required for them, and the notebook prints their real columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# src/forecast/data_loader.py -> project root is two levels above this folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATE_FORMAT = "%Y-%m-%d"  # explicit format: avoids the pandas "Could not infer format" warning


def load_config(config_path: Optional[Path] = None) -> dict:
    """Read config.yaml and return it as a dictionary.

    Input : optional path to a YAML file (default: <project root>/config.yaml).
    Output: dict with the keys paths / data / forecast / split / model.
    """
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_raw_path(filename: str, config: Optional[dict] = None) -> Path:
    """Return the full path of a file inside data/raw (folder read from the config)."""
    config = config or load_config()
    path = PROJECT_ROOT / config["paths"]["raw_dir"] / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{filename} not found in {path.parent}. "
            "Run setup_project.py with --source, or copy the CSV there."
        )
    return path


def validate_columns(df: pd.DataFrame, required: list, name: str) -> None:
    """Raise a clear error if some expected columns are missing (never guess columns)."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: missing columns {missing}. Columns found: {list(df.columns)}"
        )


def _read_csv(filename: str, config: dict, date_col: Optional[str] = "date", **kwargs) -> pd.DataFrame:
    """Read one raw CSV and convert its date column with an explicit format."""
    df = pd.read_csv(get_raw_path(filename, config), **kwargs)
    if date_col:
        validate_columns(df, [date_col], filename)
        df[date_col] = pd.to_datetime(df[date_col], format=DATE_FORMAT, errors="raise")
    return df


def load_train(config: Optional[dict] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load train.csv (3,000,888 rows): daily sales per store and product family.

    Input : config dict (default: config.yaml), nrows to read only a sample.
    Output: DataFrame with `date` as datetime64 and `family` as category.
    """
    config = config or load_config()
    d = config["data"]
    df = _read_csv(
        d["train_file"], config, date_col=d["date_col"], nrows=nrows,
        dtype={"id": "int32", "store_nbr": "int16", d["product_col"]: "category",
               d["promo_col"]: "int32"},
    )
    validate_columns(df, [d["date_col"], "store_nbr", d["product_col"], d["target_col"], d["promo_col"]], d["train_file"])
    return df


def load_stores(config: Optional[dict] = None) -> pd.DataFrame:
    """Load stores.csv (store_nbr, city, state, type, cluster)."""
    config = config or load_config()
    df = _read_csv("stores.csv", config, date_col=None)
    validate_columns(df, ["store_nbr", "city", "state", "type", "cluster"], "stores.csv")
    return df


def load_holidays(config: Optional[dict] = None) -> pd.DataFrame:
    """Load holidays_events.csv; `transferred` is cast to bool."""
    config = config or load_config()
    df = _read_csv("holidays_events.csv", config)
    validate_columns(df, ["type", "locale", "locale_name", "description", "transferred"], "holidays_events.csv")
    df["transferred"] = df["transferred"].astype(bool)
    return df


def load_oil(config: Optional[dict] = None) -> pd.DataFrame:
    """Load oil.csv (only `date` is required: other columns not inspected yet)."""
    return _read_csv("oil.csv", config or load_config())


def load_transactions(config: Optional[dict] = None) -> pd.DataFrame:
    """Load transactions.csv (only `date` is required: other columns not inspected yet)."""
    return _read_csv("transactions.csv", config or load_config())


def find_missing_dates(dates: pd.Series) -> pd.DatetimeIndex:
    """Return the calendar days that never appear between the first and last date.

    Input : a datetime Series (e.g. train['date']).
    Output: DatetimeIndex of the missing days (expected: Dec 25 of each year).
    """
    unique = pd.DatetimeIndex(pd.Series(dates).dropna().unique()).sort_values()
    if len(unique) == 0:
        return pd.DatetimeIndex([])
    full = pd.date_range(unique.min(), unique.max(), freq="D")
    return full.difference(unique)


def check_panel_completeness(df: pd.DataFrame, config: Optional[dict] = None) -> dict:
    """Check that every (date, store, family) combination exists exactly once.

    Output: dict with n_dates, n_stores, n_families, expected_rows, actual_rows,
    is_complete, duplicated_keys.
    """
    config = config or load_config()
    d = config["data"]
    n_dates = df[d["date_col"]].nunique()
    n_stores = df["store_nbr"].nunique()
    n_fam = df[d["product_col"]].nunique()
    keys = [d["date_col"], "store_nbr", d["product_col"]]
    expected = n_dates * n_stores * n_fam
    n_dup = int(df.duplicated(subset=keys).sum())
    return {
        "n_dates": int(n_dates), "n_stores": int(n_stores), "n_families": int(n_fam),
        "expected_rows": int(expected), "actual_rows": int(len(df)),
        "is_complete": bool(n_dup == 0 and expected == len(df)),
        "duplicated_keys": n_dup,
    }
