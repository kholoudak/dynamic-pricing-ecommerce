"""
Step 8 - Prediction function / Etape 8 - Fonction de prediction.

Run:  python -m src.forecast.predict                                   (summary of all products)
      python -m src.forecast.predict --product "GROCERY I" --week 1    (one value)

    from src.forecast.predict import predict_demand
    predict_demand("GROCERY I", 1)            # next week
    predict_demand("GROCERY I", "2017-08-21") # a date works too (any day of the target week)

What it does
  - the forecast ORIGIN is the last week of the data (outputs/sales_weekly.csv);
  - the input of the networks is the last `window` weeks of features of each product, built exactly
    like in training (same features, same scaler fitted on the training weeks, same seasonal input);
  - final forecast (config `prediction`):  (1 - w) * mean(LSTM ensemble) + w * seasonal naive,
    with w = prediction.seasonal_weight (0.5 = hybrid chosen on validation, 0.0 = pure LSTM ensemble);
  - result in ORIGINAL units (units sold per week), never negative.

Config (optional, defaults shown):
    prediction:
      use_ensemble: true      # models/ensemble_meta.json (Step 6d); false = single model of Step 6
      seasonal_weight: 0.5    # 0.0 = pure LSTM, 0.5 = hybrid
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from src.forecast import data_loader as dl
from src.forecast import ensemble as en
from src.forecast import evaluate as ev
from src.forecast import sequence_generator as sg

logger = logging.getLogger(__name__)
PREDICTION_DEFAULTS = {"use_ensemble": True, "seasonal_weight": 0.5}


def get_prediction_params(config: dict) -> dict:
    """`prediction` settings of config.yaml with defaults; seasonal_weight must be in [0, 1]."""
    params = {**PREDICTION_DEFAULTS, **(config.get("prediction") or {})}
    w = float(params["seasonal_weight"])
    if not 0.0 <= w <= 1.0:
        raise ValueError("prediction.seasonal_weight must be between 0 and 1.")
    params["seasonal_weight"] = w
    params["use_ensemble"] = bool(params["use_ensemble"])
    return params


def to_week_start(value) -> pd.Timestamp:
    """Any date -> the Monday of its week (the weekly tables use Monday-start weeks)."""
    ts = pd.Timestamp(value).normalize()
    return ts - pd.Timedelta(days=int(ts.day_of_week))


def build_inference_inputs(features: pd.DataFrame, scaler: sg.FeatureScaler, feature_cols: List[str],
                           window: int, horizon: int, product_index: dict, season_length: int = 52) -> dict:
    """Build ONE sample per product: the last `window` weeks of the data (origin = last week).

    Input : features = unscaled output of Step 5a (data/processed/features_weekly.csv).
    Output: {"X", "S", "meta"} in the same format as the training sequences (no target).
    Raises ValueError if a product does not end at the same week as the others or has a gap.
    """
    scaled = scaler.transform(features)
    last_week = scaled["week"].max()
    xs, ss, rows = [], [], []
    for pid, g in scaled.sort_values("week").groupby("product_id"):
        weeks = pd.DatetimeIndex(g["week"])
        if weeks[-1] != last_week:
            raise ValueError(f"{pid}: the series ends on {weeks[-1].date()} but the data ends on {last_week.date()}.")
        if len(g) < window:
            raise ValueError(f"{pid}: only {len(g)} weeks of features, {window} needed.")
        tail = g.tail(window)
        if not (pd.DatetimeIndex(tail["week"])[1:] - pd.DatetimeIndex(tail["week"])[:-1] == pd.Timedelta(days=7)).all():
            raise ValueError(f"{pid}: the last {window} weeks are not consecutive.")
        zmap = dict(zip(weeks, g["log_sales"].to_numpy(float)))
        xs.append(tail[feature_cols].to_numpy(np.float32))
        ss.append(sg.seasonal_features(zmap, last_week, horizon, season_length))
        rows.append((str(pid), product_index[str(pid)], last_week))
    meta = pd.DataFrame(rows, columns=["product_id", "product_idx", "origin_week"])
    return {"X": np.stack(xs).astype(np.float32), "S": np.stack(ss).astype(np.float32), "meta": meta}


class DemandForecaster:
    """Loads everything once and produces the forecasts of every product for weeks +1 .. +horizon."""

    def __init__(self, config: Optional[dict] = None, root: Optional[Path] = None):
        from src.forecast import model as md
        from src.forecast.train import load_model_meta

        self.config = config or dl.load_config()
        self.root = Path(root) if root else dl.PROJECT_ROOT
        self.params = get_prediction_params(self.config)
        paths = self.config["paths"]
        models_dir, outputs_dir, proc_dir = (self.root / paths[k] for k in ("models_dir", "outputs_dir", "processed_dir"))

        weekly_path = outputs_dir / self.config.get("preprocessing", {}).get("weekly_file", "sales_weekly.csv")
        feats_path = proc_dir / "features_weekly.csv"
        for p in (weekly_path, feats_path, models_dir / "feature_scaler.json", outputs_dir / "split_info.json"):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found: run Steps 4-5 first.")
        self.weekly = pd.read_csv(weekly_path, parse_dates=["week"])
        self.weekly["product_id"] = self.weekly["product_id"].astype(str)
        features = pd.read_csv(feats_path, parse_dates=["week"])
        features["product_id"] = features["product_id"].astype(str)
        self.scaler = sg.FeatureScaler.load(models_dir / "feature_scaler.json")
        info = json.loads((outputs_dir / "split_info.json").read_text(encoding="utf-8"))
        self.horizon, self.window = int(info["horizon"]), int(info["window_size"])
        self.products: List[str] = list(info["products"])

        self.members = self._load_members(models_dir, md, load_model_meta)
        self.inputs = build_inference_inputs(features, self.scaler, info["feature_cols"], self.window, self.horizon,
                                             {p: i for i, p in enumerate(self.products)},
                                             self.config["forecast"].get("season_length", 52))
        self.origin_week = pd.Timestamp(self.inputs["meta"]["origin_week"].iloc[0])
        self._table: Optional[pd.DataFrame] = None

    def _load_members(self, models_dir: Path, md, load_model_meta) -> List[dict]:
        if self.params["use_ensemble"]:
            meta_path = models_dir / "ensemble_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"{meta_path} not found: run `python -m src.forecast.ensemble` "
                                        "or set prediction.use_ensemble: false.")
            em = json.loads(meta_path.read_text(encoding="utf-8"))
            pairs = list(zip(em["files"], em["meta_files"]))
        else:
            stem_file = Path(self.config["model"].get("model_file", "lstm_demand_model.keras")).name
            pairs = [(stem_file, "model_meta.json")]
        return [{"model": md.load_model(models_dir / f), "model_meta": load_model_meta(models_dir / m)} for f, m in pairs]

    def forecast_table(self) -> pd.DataFrame:
        """Forecast of every product and horizon: product_id, origin_week, horizon, week, demand_pred (+ components)."""
        if self._table is not None:
            return self._table
        seq, meta = self.inputs, self.inputs["meta"]
        lstm = en.average_forecasts(en.predict_members(self.members, seq, self.scaler))
        lookup = ev.build_sales_lookup(self.weekly)
        ep = ev.get_eval_params(self.config)
        seasonal = ev.baseline_forecasts(lookup, meta, self.horizon, ep["ma_window"], ep["season_length"])["seasonal_naive"]
        w = self.params["seasonal_weight"]
        final = np.clip(en.blend(lstm, seasonal, 1.0 - w), 0, None)
        rows = []
        for i, pid in enumerate(meta["product_id"]):
            for h in range(1, self.horizon + 1):
                rows.append({"product_id": pid, "origin_week": self.origin_week, "horizon": h,
                             "week": self.origin_week + pd.Timedelta(days=7 * h),
                             "demand_pred": float(final[i, h - 1]),
                             "lstm_pred": float(lstm[i, h - 1]), "seasonal_naive_pred": float(seasonal[i, h - 1])})
        self._table = pd.DataFrame(rows).sort_values(["product_id", "horizon"]).reset_index(drop=True)
        return self._table

    def predict(self, product_id: str, week: Union[int, str, pd.Timestamp]) -> float:
        """Forecast (units/week) of one product for week +1..+horizon (int) or for the week containing a date."""
        if product_id not in self.products:
            raise ValueError(f"Unknown product_id {product_id!r}. Known products: {self.products[:5]} ... "
                             f"({len(self.products)} in total).")
        table = self.forecast_table()
        if isinstance(week, (int, np.integer)) and not isinstance(week, bool):
            if not 1 <= int(week) <= self.horizon:
                raise ValueError(f"week must be between 1 and {self.horizon} (weeks ahead), got {week}.")
            row = table[(table["product_id"] == product_id) & (table["horizon"] == int(week))]
        else:
            try:
                start = to_week_start(week)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Cannot read week={week!r}: use 1..{self.horizon} or a date.") from exc
            row = table[(table["product_id"] == product_id) & (table["week"] == start)]
            if row.empty:
                first, last = self.origin_week + pd.Timedelta(days=7), self.origin_week + pd.Timedelta(days=7 * self.horizon)
                raise ValueError(f"Week {start.date()} is outside the forecast range {first.date()} .. {last.date()}.")
        return float(row["demand_pred"].iloc[0])


_DEFAULT: Optional[DemandForecaster] = None


def predict_demand(product_id: str, week: Union[int, str, pd.Timestamp]) -> float:
    """Forecast weekly demand (units) of `product_id` for `week`.

    week: 1..4 = number of weeks after the last week of the data, or a date (the week containing it).
    The models are loaded on the first call and kept in memory.
    Raises ValueError for an unknown product or a week outside the 4-week range.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = DemandForecaster()
    return _DEFAULT.predict(product_id, week)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly demand forecast (1-4 weeks ahead).")
    parser.add_argument("--product", default=None)
    parser.add_argument("--week", default="1", help="1..4 or a date (YYYY-MM-DD)")
    args = parser.parse_args()
    try:
        f = DemandForecaster()
        if args.product:
            week = int(args.week) if args.week.isdigit() else args.week
            print(f"{args.product} | week {week}: {f.predict(args.product, week):,.0f} units")
            return
        table = f.forecast_table()
        print(f"Origin (last data week): {f.origin_week.date()} | w = {f.params['seasonal_weight']} | "
              f"ensemble: {f.params['use_ensemble']} ({len(f.members)} model(s))")
        wide = table.pivot(index="product_id", columns="horizon", values="demand_pred").round(0)
        wide.columns = [f"week +{c}" for c in wide.columns]
        print(wide.to_string())
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
