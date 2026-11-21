"""
Step 5b - Chronological split, scaling and sliding windows /
Etape 5b - Decoupage chronologique, normalisation et fenetres glissantes.

Rules (no data leakage)
1. The split is chronological and GLOBAL (same cut dates for every product).
2. A sample belongs to a split according to its TARGET weeks: all the weeks to predict must
   lie inside that split. Input windows may reach back into earlier splits (past history).
   Samples whose targets straddle a boundary are dropped.
3. Scalers are fitted on the training weeks only (week <= train_end) and then applied as is.
4. Sample = previous `window` weeks -> next `horizon` weeks (direct multi-step forecast):
   one model gives the forecast for 1, 2, 3 and 4 weeks ahead.

Run:  python -m src.forecast.sequence_generator
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.forecast import data_loader as dl
from src.forecast import feature_engineering as fe

logger = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- split
def compute_split_dates(weeks: pd.Series, val_weeks: int, test_weeks: int) -> dict:
    """Return the chronological cut dates from the list of weeks of the dataset.

    train: weeks <= train_end | val: train_end < week <= val_end | test: val_end < week <= test_end
    """
    uniq = pd.Series(pd.to_datetime(weeks).unique()).sort_values().reset_index(drop=True)
    if len(uniq) < val_weeks + test_weeks + 30:
        raise ValueError(f"Only {len(uniq)} weeks: too short for val={val_weeks} + test={test_weeks} + training.")
    return {
        "train_end": uniq.iloc[-(val_weeks + test_weeks) - 1],
        "val_end": uniq.iloc[-test_weeks - 1],
        "test_end": uniq.iloc[-1],
        "first_week": uniq.iloc[0],
    }


# --------------------------------------------------------------------------- scaling
class FeatureScaler:
    """Per-product standardisation fitted on training weeks only.

    - sales-derived columns (log_sales, lag_*, roll_mean_*) : (x - mu_sales) / sd_sales
    - roll_std_*                                            : x / sd_sales
    - log_promo                                             : (x - mu_promo) / sd_promo (per product)
    - open_stores                                           : (x - mu_open) / sd_open (global)
    - other columns (calendar, flags, holiday counts)       : unchanged
    The scaled `log_sales` is also the TARGET; `inverse_sales` goes back to units.
    """

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = list(feature_cols)
        self.params: Dict[str, dict] = {}
        self.open_stores = {"mu": 0.0, "sd": 1.0}

    @staticmethod
    def _safe_sd(x: pd.Series) -> float:
        sd = float(x.std(ddof=0))
        return sd if np.isfinite(sd) and sd > 1e-8 else 1.0

    def fit(self, train_df: pd.DataFrame) -> "FeatureScaler":
        """Learn mean / std on the training rows only."""
        for pid, g in train_df.groupby("product_id"):
            self.params[str(pid)] = {
                "mu_sales": float(g["log_sales"].mean()), "sd_sales": self._safe_sd(g["log_sales"]),
                "mu_promo": float(g["log_promo"].mean()), "sd_promo": self._safe_sd(g["log_promo"]),
            }
        self.open_stores = {"mu": float(train_df["open_stores"].mean()), "sd": self._safe_sd(train_df["open_stores"])}
        return self

    def _per_product(self, df: pd.DataFrame, key: str) -> np.ndarray:
        unknown = set(df["product_id"].astype(str)) - set(self.params)
        if unknown:
            raise ValueError(f"Products unseen when the scaler was fitted: {sorted(unknown)[:5]}")
        return df["product_id"].astype(str).map({k: v[key] for k, v in self.params.items()}).to_numpy(float)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a scaled copy of `df` (the original is not modified)."""
        if not self.params:
            raise RuntimeError("Scaler not fitted.")
        out = df.copy()
        mu_s, sd_s = self._per_product(df, "mu_sales"), self._per_product(df, "sd_sales")
        for col in self.feature_cols:
            if col == "log_sales" or col.startswith(("lag_", "roll_mean_")):
                out[col] = (df[col].to_numpy(float) - mu_s) / sd_s
            elif col.startswith("roll_std_"):
                out[col] = df[col].to_numpy(float) / sd_s
            elif col == "log_promo":
                out[col] = (df[col].to_numpy(float) - self._per_product(df, "mu_promo")) / self._per_product(df, "sd_promo")
            elif col == "open_stores":
                out[col] = (df[col].to_numpy(float) - self.open_stores["mu"]) / self.open_stores["sd"]
        return out

    def inverse_sales(self, product_ids, y_scaled) -> np.ndarray:
        """Scaled log-sales -> weekly sales in ORIGINAL units (>= 0).

        product_ids: length n; y_scaled: shape (n,) or (n, horizon).
        """
        ids = pd.Series(np.asarray(product_ids).astype(str))
        mu = ids.map({k: v["mu_sales"] for k, v in self.params.items()}).to_numpy(float)
        sd = ids.map({k: v["sd_sales"] for k, v in self.params.items()}).to_numpy(float)
        y = np.asarray(y_scaled, dtype=float)
        if y.ndim == 2:
            mu, sd = mu[:, None], sd[:, None]
        return np.clip(np.expm1(y * sd + mu), 0, None)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"feature_cols": self.feature_cols, "params": self.params, "open_stores": self.open_stores}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        s = cls(d["feature_cols"])
        s.params, s.open_stores = d["params"], d["open_stores"]
        return s


# --------------------------------------------------------------------------- windows
def assign_split(first_target: pd.Timestamp, last_target: pd.Timestamp, split_dates: dict) -> Optional[str]:
    """Split of a sample given its first and last TARGET week (None if it straddles a boundary)."""
    if last_target <= split_dates["train_end"]:
        return "train"
    if first_target > split_dates["train_end"] and last_target <= split_dates["val_end"]:
        return "val"
    if first_target > split_dates["val_end"]:
        return "test"
    return None


def seasonal_features(zmap: Dict[pd.Timestamp, float], origin_week: pd.Timestamp, horizon: int,
                      season_length: int = 52) -> np.ndarray:
    """What happened ONE YEAR AGO between the origin week and each target week.

    Returns [s_1 .. s_horizon, has_seasonal] with s_h = z(origin + h - 52 weeks) - z(origin - 52 weeks),
    z = scaled log-sales. Only weeks <= origin are read (h <= 52), so nothing comes from the future.
    If last year's weeks do not exist (start of a series) the vector is zeros with has_seasonal = 0.
    """
    shift = pd.Timedelta(days=7 * season_length)
    base = zmap.get(origin_week - shift)
    out = np.zeros(horizon + 1, dtype=np.float32)
    if base is None:
        return out
    for h in range(1, horizon + 1):
        value = zmap.get(origin_week + pd.Timedelta(days=7 * h) - shift)
        if value is None:
            return np.zeros(horizon + 1, dtype=np.float32)
        out[h - 1] = value - base
    out[horizon] = 1.0
    return out


def make_sequences(scaled: pd.DataFrame, feature_cols: List[str], window: int, horizon: int,
                   split_dates: dict, product_index: Dict[str, int], season_length: int = 52) -> Dict[str, dict]:
    """Slide a window over each product: X = `window` weeks of features, y = next `horizon` log-sales.

    Output: {"train"|"val"|"test": {"X": (n, window, n_features), "y": (n, horizon),
             "S": (n, horizon + 1) seasonal reference, "meta": DataFrame}}
    meta columns: product_id, product_idx, origin_week (last input week), first_target_week, last_target_week.
    """
    buckets = {s: {"X": [], "y": [], "S": [], "meta": []} for s in SPLITS}
    for pid, g in scaled.sort_values("week").groupby("product_id"):
        weeks = pd.DatetimeIndex(g["week"])
        if len(weeks) > 1 and not (weeks[1:] - weeks[:-1] == pd.Timedelta(days=7)).all():
            raise ValueError(f"{pid}: weeks are not consecutive.")
        feats = g[feature_cols].to_numpy(np.float32)
        target = g["log_sales"].to_numpy(np.float32)
        zmap = dict(zip(weeks, target.astype(float)))
        for i in range(window - 1, len(g) - horizon):
            first_t, last_t = weeks[i + 1], weeks[i + horizon]
            split = assign_split(first_t, last_t, split_dates)
            if split is None:
                continue
            b = buckets[split]
            b["X"].append(feats[i - window + 1: i + 1])
            b["y"].append(target[i + 1: i + 1 + horizon])
            b["S"].append(seasonal_features(zmap, weeks[i], horizon, season_length))
            b["meta"].append((str(pid), product_index[str(pid)], weeks[i], first_t, last_t))

    cols = ["product_id", "product_idx", "origin_week", "first_target_week", "last_target_week"]
    out = {}
    for s, b in buckets.items():
        n_feat = len(feature_cols)
        out[s] = {
            "X": np.stack(b["X"]).astype(np.float32) if b["X"] else np.zeros((0, window, n_feat), np.float32),
            "y": np.stack(b["y"]).astype(np.float32) if b["y"] else np.zeros((0, horizon), np.float32),
            "S": np.stack(b["S"]).astype(np.float32) if b["S"] else np.zeros((0, horizon + 1), np.float32),
            "meta": pd.DataFrame(b["meta"], columns=cols),
        }
    return out


# --------------------------------------------------------------------------- pipeline
def prepare_dataset(config: Optional[dict] = None, weekly: Optional[pd.DataFrame] = None,
                    holidays: Optional[pd.DataFrame] = None, out_root: Optional[Path] = None,
                    save: bool = True) -> dict:
    """Features -> split -> scaler (train only) -> sliding windows -> files.

    Writes (under out_root, default = project root):
      data/processed/features_weekly.csv, data/processed/sequences.npz,
      data/processed/sequence_meta_{train,val,test}.csv,
      models/feature_scaler.json, outputs/split_info.json
    """
    config = config or dl.load_config()
    root = Path(out_root) if out_root else dl.PROJECT_ROOT
    if weekly is None:
        path = root / config["paths"]["outputs_dir"] / config.get("preprocessing", {}).get("weekly_file", "sales_weekly.csv")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run Step 4 first (python -m src.forecast.preprocessing).")
        weekly = pd.read_csv(path, parse_dates=["week"])
    if holidays is None:
        holidays = dl.load_holidays(config)

    window, horizon = config["forecast"]["window_size"], config["forecast"]["horizon"]
    feats, feature_cols = fe.build_features(weekly, holidays, config)
    split_dates = compute_split_dates(feats["week"], config["split"]["val_weeks"], config["split"]["test_weeks"])

    scaler = FeatureScaler(feature_cols).fit(feats[feats["week"] <= split_dates["train_end"]])
    scaled = scaler.transform(feats)
    products = sorted(feats["product_id"].astype(str).unique())
    product_index = {p: i for i, p in enumerate(products)}
    seq = make_sequences(scaled, feature_cols, window, horizon, split_dates, product_index,
                         config["forecast"].get("season_length", 52))

    info = {
        "window_size": window, "horizon": horizon, "feature_cols": feature_cols, "products": products,
        "split_dates": {k: pd.Timestamp(v).date().isoformat() for k, v in split_dates.items()},
        "n_samples": {s: int(len(seq[s]["X"])) for s in SPLITS},
        "n_weeks_after_feature_drop": int(feats["week"].nunique()),
    }
    if save:
        proc, models, outputs = (root / config["paths"][k] for k in ("processed_dir", "models_dir", "outputs_dir"))
        for d in (proc, models, outputs):
            d.mkdir(parents=True, exist_ok=True)
        feats.to_csv(proc / "features_weekly.csv", index=False)
        np.savez_compressed(proc / "sequences.npz", **{f"{k}_{s}": seq[s][k] for s in SPLITS for k in ("X", "y", "S")})
        for s in SPLITS:
            seq[s]["meta"].to_csv(proc / f"sequence_meta_{s}.csv", index=False)
        scaler.save(models / "feature_scaler.json")
        (outputs / "split_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {"features": feats, "scaled": scaled, "sequences": seq, "scaler": scaler,
            "split_dates": split_dates, "feature_cols": feature_cols, "info": info}


def load_prepared(config: Optional[dict] = None, root: Optional[Path] = None) -> dict:
    """Reload what `prepare_dataset` saved (used by the training step)."""
    config = config or dl.load_config()
    root = Path(root) if root else dl.PROJECT_ROOT
    proc = root / config["paths"]["processed_dir"]
    arrays = np.load(proc / "sequences.npz")
    seq = {s: {"X": arrays[f"X_{s}"], "y": arrays[f"y_{s}"],
               "S": arrays[f"S_{s}"] if f"S_{s}" in arrays.files else None,
               "meta": pd.read_csv(proc / f"sequence_meta_{s}.csv", parse_dates=["origin_week", "first_target_week", "last_target_week"])}
           for s in SPLITS}
    info = json.loads((root / config["paths"]["outputs_dir"] / "split_info.json").read_text(encoding="utf-8"))
    scaler = FeatureScaler.load(root / config["paths"]["models_dir"] / "feature_scaler.json")
    return {"sequences": seq, "scaler": scaler, "info": info}


def plot_split(features: pd.DataFrame, split_dates: dict, path: Path) -> None:
    """Save a chart of the total weekly sales with the train / val / test regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = features.groupby("week")["sales"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(total.index, total.values, linewidth=1)
    ax.axvspan(total.index.min(), split_dates["train_end"], color="tab:blue", alpha=0.08, label="train")
    ax.axvspan(split_dates["train_end"], split_dates["val_end"], color="tab:orange", alpha=0.15, label="validation")
    ax.axvspan(split_dates["val_end"], split_dates["test_end"], color="tab:green", alpha=0.15, label="test")
    ax.set_title("Chronological split (total weekly sales of the kept products)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = dl.load_config()
    res = prepare_dataset(cfg)
    info = res["info"]
    print("Products:", len(info["products"]), "| features:", len(info["feature_cols"]))
    print("Feature columns:", info["feature_cols"])
    print("Split dates:", info["split_dates"])
    print("Samples:", info["n_samples"])
    for s in SPLITS:
        print(f"X_{s}: {res['sequences'][s]['X'].shape} | y_{s}: {res['sequences'][s]['y'].shape}")
    plot_split(res["features"], res["split_dates"], dl.PROJECT_ROOT / cfg["paths"]["plots_dir"] / "05_train_val_test_split.png")
    print("Saved: data/processed/*, models/feature_scaler.json, outputs/split_info.json, outputs/plots/05_train_val_test_split.png")
