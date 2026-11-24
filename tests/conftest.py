"""Shared test fixtures (Agent 3)."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.pricing.config import load_config


@pytest.fixture
def mock_cfg(tmp_path):
    rng = np.random.default_rng(3)
    fams = ["A", "B", "C"]
    fc_weeks = pd.date_range("2017-08-14", periods=4, freq="W-MON")
    pd.DataFrame([{"product_id": f, "week": w, "demand_pred": 1000.0 * (i + 1) + 10 * k, "horizon": k + 1}
                  for i, f in enumerate(fams) for k, w in enumerate(fc_weeks)]).to_csv(tmp_path / "forecast.csv", index=False)

    hist_weeks = pd.date_range("2015-01-05", periods=130, freq="W-MON")
    pd.DataFrame([{"product_id": f, "week": w, "sales": 1000.0 * (i + 1) * rng.uniform(0.9, 1.1),
                   "onpromotion": float(rng.integers(0, 20)), "is_outlier": False}
                  for i, f in enumerate(fams) for w in hist_weeks]).to_csv(tmp_path / "sales.csv", index=False)

    pd.DataFrame({"product_name": [f"prod {i}" for i in range(6)], "reputation_score": rng.uniform(0.4, 0.9, 6),
                  "n_reviews": [100] * 6}).to_csv(tmp_path / "reputation.csv", index=False)

    price = rng.uniform(20, 200, 9)
    pd.DataFrame({"product_id": range(1, 10), "current_price": price, "competitor_price": price * 1.03,
                  "product_cost": price * rng.uniform(0.55, 0.85, 9)}).to_csv(tmp_path / "products.csv", index=False)

    cfg = copy.deepcopy(load_config())
    cfg["paths"].update(forecast_file=str(tmp_path / "forecast.csv"), sales_file=str(tmp_path / "sales.csv"),
                        reputation_file=str(tmp_path / "reputation.csv"), products_file=str(tmp_path / "products.csv"),
                        mapping_file=str(tmp_path / "mapping.csv"), master_file=str(tmp_path / "master.csv"),
                        recommendations_file=str(tmp_path / "out" / "recommended_prices.csv"),
                        reports_dir=str(tmp_path / "reports"))
    return cfg
