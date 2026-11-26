"""Price-demand regression (Agent 3, Phase 3).

Two separate analyses, deliberately NOT mixed:

1) ``fit_elasticity_model``  - log-log price elasticity regression
       log(demand) = b0 + b1*log(price/usual price) + b2*log(competitor/usual competitor)
                     + b3*reputation + b4*log(baseline demand) + error
   b1 is the price elasticity. It needs historical prices, which the real data does not have,
   so in this project it is run on SIMULATED data (see synthetic.py) to validate the method.

2) ``fit_promo_effect``  - what the REAL sales data does contain: promotions.
   It measures how sales move with the number of items on promotion. This is a promotion
   effect, NOT a price elasticity.

Validation is always chronological (train on the past, test on the future).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

FEATURES = ["log_price_rel", "log_comp_rel", "reputation_score", "log_baseline_demand"]


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------

def chronological_split(panel, test_fraction):
    """Split by WEEK: the last ``test_fraction`` of the weeks form the test set."""
    weeks = np.sort(panel["week"].unique())
    n_train = int(round(len(weeks) * (1 - test_fraction)))
    n_train = min(max(n_train, 1), len(weeks) - 1)
    train_weeks = weeks[:n_train]
    train = panel[panel["week"].isin(train_weeks)]
    test = panel[~panel["week"].isin(train_weeks)]
    return train.copy(), test.copy()


def add_features(panel, ref=None):
    """Build the regression columns.

    Prices are measured RELATIVE to each product's usual (average) log price, so a cheap and an
    expensive product are compared fairly. ``ref`` must come from the TRAINING data only
    (otherwise information from the future leaks into the model).
    Returns (table with features, ref).
    """
    out = panel.copy()
    out["_lp"] = np.log(out["price"])
    out["_lc"] = np.log(out["competitor_price_t"])
    if ref is None:
        ref = out.groupby("product_id")[["_lp", "_lc"]].mean().add_prefix("ref")
    out = out.join(ref, on="product_id")
    out["log_price_rel"] = out["_lp"] - out["ref_lp"]
    out["log_comp_rel"] = out["_lc"] - out["ref_lc"]
    out["log_baseline_demand"] = np.log(out["baseline_demand"])
    out["y"] = np.log(out["demand"])
    return out.drop(columns=["_lp", "_lc", "ref_lp", "ref_lc"]), ref


def _metrics(y_true_log, y_pred_log, base_log=None):
    """R2, MAE, RMSE on the log scale and on the demand scale.

    R2_log looks very high because products differ hugely in size. If ``base_log`` (the
    log baseline demand) is given we also report R2_vs_baseline: how much of what is LEFT after
    removing the baseline does the model explain? That is the honest measure of its value.
    """
    extra = {}
    if base_log is not None:
        extra["R2_vs_baseline"] = float(r2_score(y_true_log - base_log, y_pred_log - base_log))
    return {**extra,
        "R2_log": float(r2_score(y_true_log, y_pred_log)),
        "MAE_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "MAE_demand": float(mean_absolute_error(np.exp(y_true_log), np.exp(y_pred_log))),
        "RMSE_demand": float(np.sqrt(mean_squared_error(np.exp(y_true_log), np.exp(y_pred_log)))),
    }


def compute_vif(X):
    """Variance Inflation Factor of each feature (> 10 is the usual warning sign)."""
    Xc = sm.add_constant(X, has_constant="add")
    return {col: float(variance_inflation_factor(Xc.values, i))
            for i, col in enumerate(Xc.columns) if col != "const"}


@dataclass
class LinearFit:
    """A fitted linear model stored as plain numbers (works for OLS and Ridge alike)."""
    kind: str                  # "OLS" or "Ridge"
    intercept: float
    coef: dict
    alpha: float = None        # Ridge penalty (None for OLS)
    p_values: dict = field(default_factory=dict)
    conf_int: dict = field(default_factory=dict)

    def predict_log(self, X):
        """Predicted log(demand) for a table of features."""
        return self.intercept + sum(self.coef[c] * X[c] for c in self.coef)


def _fit(train, use_ridge, alphas):
    """Fit OLS (statsmodels, gives p-values) or Ridge (scikit-learn, handles collinearity)."""
    X, y = train[FEATURES], train["y"]
    if not use_ridge:
        res = sm.OLS(y, sm.add_constant(X)).fit()
        ci = res.conf_int()
        return LinearFit("OLS", float(res.params["const"]),
                         {c: float(res.params[c]) for c in FEATURES},
                         p_values={c: float(res.pvalues[c]) for c in FEATURES},
                         conf_int={c: (float(ci.loc[c, 0]), float(ci.loc[c, 1])) for c in FEATURES})
    scaler = StandardScaler().fit(X)
    # rows are ordered by week, so TimeSeriesSplit keeps the past before the future
    ridge = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(3)).fit(scaler.transform(X), y)
    coef = ridge.coef_ / scaler.scale_                       # back to the original units
    intercept = ridge.intercept_ - float(np.sum(coef * scaler.mean_))
    return LinearFit("Ridge", float(intercept), dict(zip(FEATURES, map(float, coef))),
                     alpha=float(ridge.alpha_))


# ----------------------------------------------------------------------------------------
# 1) Elasticity model
# ----------------------------------------------------------------------------------------

@dataclass
class ElasticityResult:
    """Everything we report about the elasticity regression."""
    model: LinearFit                 # final model, refitted on all weeks
    ref: pd.DataFrame                # usual log prices per product (from ALL weeks)
    elasticity: float
    vif: dict
    used_ridge: bool
    holdout_metrics: dict
    baseline_metrics: dict           # "no price information" reference
    cv_table: pd.DataFrame
    n_train_weeks: int
    n_test_weeks: int
    notes: list = field(default_factory=list)

    def predict_demand(self, price, competitor_price, reputation, baseline_demand, product_id):
        """Predicted demand at a given price (used by the optimizer in Phase 4)."""
        row = pd.DataFrame({"product_id": [product_id], "price": [price],
                            "competitor_price_t": [competitor_price], "reputation_score": [reputation],
                            "baseline_demand": [baseline_demand], "demand": [1.0]})
        feats, _ = add_features(row, ref=self.ref)
        return float(np.exp(self.model.predict_log(feats).iloc[0]))


def fit_elasticity_model(panel, reg_cfg):
    """Fit the log-log elasticity model with chronological validation.

    Steps: split by time -> check multicollinearity (VIF) on training data -> OLS, or Ridge if
    max VIF > threshold -> evaluate on the held-out FUTURE weeks -> temporal cross-validation ->
    refit on all weeks for use downstream.
    """
    panel = panel.sort_values(["week", "product_id"]).reset_index(drop=True)
    alphas = reg_cfg["ridge_alphas"]
    notes = []

    train_raw, test_raw = chronological_split(panel, reg_cfg["test_fraction"])
    train, ref_tr = add_features(train_raw)
    test, _ = add_features(test_raw, ref=ref_tr)            # test uses the TRAIN reference

    vif = compute_vif(train[FEATURES])
    use_ridge = max(vif.values()) > reg_cfg["vif_threshold"]
    notes.append(f"max VIF = {max(vif.values()):.2f} (threshold {reg_cfg['vif_threshold']}) -> "
                 f"{'Ridge used (multicollinearity)' if use_ridge else 'plain OLS used'}.")

    fit_tr = _fit(train, use_ridge, alphas)
    holdout = _metrics(test["y"], fit_tr.predict_log(test), test["log_baseline_demand"])
    baseline = _metrics(test["y"], test["log_baseline_demand"])   # "demand = Agent 1 forecast"

    # expanding-window temporal cross-validation over weeks
    weeks = np.sort(panel["week"].unique())
    rows = []
    for k, (tr_idx, te_idx) in enumerate(TimeSeriesSplit(reg_cfg["n_cv_splits"]).split(weeks), 1):
        tr_k, ref_k = add_features(panel[panel["week"].isin(weeks[tr_idx])])
        te_k, _ = add_features(panel[panel["week"].isin(weeks[te_idx])], ref=ref_k)
        m = _metrics(te_k["y"], _fit(tr_k, use_ridge, alphas).predict_log(te_k), te_k["log_baseline_demand"])
        rows.append({"fold": k, "train_weeks": len(tr_idx), "test_weeks": len(te_idx), **m})
    cv_table = pd.DataFrame(rows)

    # final model on all weeks
    full, ref_full = add_features(panel)
    final = _fit(full, use_ridge, alphas)

    return ElasticityResult(final, ref_full, final.coef["log_price_rel"], vif, use_ridge, holdout,
                            baseline, cv_table, int(train_raw["week"].nunique()),
                            int(test_raw["week"].nunique()), notes)


# ----------------------------------------------------------------------------------------
# 2) Promotion effect on REAL sales (descriptive, not a price elasticity)
# ----------------------------------------------------------------------------------------

def fit_promo_effect(sales, test_weeks=52):
    """Relate real weekly sales to promotion intensity, per product family.

    log(sales) = a_family + b*log(1 + items on promotion) + seasonality + error
    (a_family handled by subtracting each family's TRAIN average). Last ``test_weeks`` are held out.
    """
    s = sales[(sales["sales"] > 0)].copy()
    if "is_outlier" in s.columns:
        s = s[~s["is_outlier"].astype(bool)]
    s["y"] = np.log(s["sales"])
    s["promo"] = np.log1p(s["onpromotion"])
    woy = s["week"].dt.isocalendar().week.astype(int)
    s["sin"] = np.sin(2 * np.pi * woy / 52)
    s["cos"] = np.cos(2 * np.pi * woy / 52)

    cut = s["week"].max() - pd.Timedelta(weeks=test_weeks)
    train, test = s[s["week"] <= cut], s[s["week"] > cut]
    cols = ["y", "promo", "sin", "cos"]
    mu = train.groupby("product_id")[cols].mean().add_suffix("_mu")

    def centre(df):
        """Subtract each family's TRAIN average (removes family-level differences)."""
        j = df[["product_id"]].join(mu, on="product_id")
        return pd.DataFrame({c: df[c] - j[c + "_mu"] for c in cols}, index=df.index).dropna()

    w_tr, w_te = centre(train), centre(test)
    X = ["promo", "sin", "cos"]
    res = sm.OLS(w_tr["y"], w_tr[X]).fit()
    pred = res.predict(w_te[X])
    return {
        "type": "REAL data - promotion effect (NOT a price elasticity)",
        "promo_coefficient": float(res.params["promo"]),
        "promo_p_value": float(res.pvalues["promo"]),
        "n_train": int(len(w_tr)), "n_test": int(len(w_te)),
        "test_R2_log": float(r2_score(w_te["y"], pred)),
        "test_MAE_log": float(mean_absolute_error(w_te["y"], pred)),
        "test_MAE_log_if_no_model": float(mean_absolute_error(w_te["y"], np.zeros(len(w_te)))),
    }


# ----------------------------------------------------------------------------------------
# Command-line report:  python -m src.pricing.regression
# ----------------------------------------------------------------------------------------

def run_regression_report(cfg=None, save=True, master=None):
    """Run both analyses on the project data and return a results dictionary.

    ``master`` (the merged table) can be passed in to avoid rebuilding it."""
    import json

    from src.pipeline.integration import run_integration
    from src.pipeline.loaders import load_sales_history
    from src.pricing.config import load_config, resolve_path
    from src.pricing.synthetic import make_synthetic_panel

    cfg = cfg or load_config()
    reg = cfg["regression"]

    if master is None:
        master = run_integration(cfg, save=False)[0].data
    panel = make_synthetic_panel(master, reg["synthetic"])
    el = fit_elasticity_model(panel, reg)
    promo = fit_promo_effect(load_sales_history(cfg).data, reg["promo"]["test_weeks"])

    results = {
        "elasticity_model": {
            "type": "SIMULATED data (known true elasticity) - validates the method only",
            "true_elasticity": reg["synthetic"]["true_elasticity"],
            "estimated_elasticity": el.elasticity,
            "model_used": el.model.kind,
            "ridge_alpha": el.model.alpha,
            "coefficients": el.model.coef,
            "p_values": el.model.p_values,
            "confidence_intervals_95": el.model.conf_int,
            "vif": el.vif,
            "train_weeks": el.n_train_weeks, "test_weeks": el.n_test_weeks,
            "holdout_metrics": el.holdout_metrics,
            "baseline_no_price_metrics": el.baseline_metrics,
            "cv_folds": el.cv_table.to_dict(orient="records"),
            "notes": el.notes,
        },
        "promotion_model": promo,
    }
    if save:
        out = resolve_path(cfg["paths"]["reports_dir"]) / "regression_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, el


if __name__ == "__main__":
    res, el = run_regression_report()
    e = res["elasticity_model"]
    print("=== 1) ELASTICITY MODEL  [SIMULATED data - true elasticity is known] ===")
    print(f"True elasticity       : {e['true_elasticity']}")
    print(f"Estimated elasticity  : {e['estimated_elasticity']:.3f}  ({e['model_used']})")
    if e["confidence_intervals_95"]:
        lo, hi = e["confidence_intervals_95"]["log_price_rel"]
        pv = e["p_values"]["log_price_rel"]
        print(f"95% interval          : [{lo:.3f}, {hi:.3f}]   p-value {'< 1e-10' if pv < 1e-10 else f'= {pv:.2g}'}")
    print("VIF                   :", {k: round(v, 2) for k, v in e["vif"].items()})
    print("Notes                 :", *e["notes"])
    print(f"Holdout ({e['test_weeks']} future weeks)  :", {k: round(v, 3) for k, v in e["holdout_metrics"].items()})
    print("Baseline (no price)   :", {k: round(v, 3) for k, v in e["baseline_no_price_metrics"].items()})
    print("Temporal CV (R2_vs_baseline per fold):", [round(f["R2_vs_baseline"], 3) for f in e["cv_folds"]])
    p = res["promotion_model"]
    print("\n=== 2) PROMOTION EFFECT  [REAL sales data - not a price elasticity] ===")
    print(f"Coefficient on log(1+items on promo): {p['promo_coefficient']:.3f}  (p = {p['promo_p_value']:.2g})")
    print(f"Test on last year: R2 = {p['test_R2_log']:.3f} | MAE = {p['test_MAE_log']:.3f} "
          f"vs {p['test_MAE_log_if_no_model']:.3f} for a model that predicts each family's average")
    print("\nSaved: reports/regression_results.json")
