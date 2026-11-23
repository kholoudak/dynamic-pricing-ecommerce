"""Tests for the integration file (run: pytest tests -v)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecast import export_forecast as xf  # noqa: E402


def good_frame(products=("A", "B"), start="2017-08-21"):
    weeks = pd.date_range(start, periods=4, freq="7D")
    rows = [(p, w.strftime("%Y-%m-%d"), 100.0 + i, i + 1) for p in products for i, w in enumerate(weeks)]
    return pd.DataFrame(rows, columns=["product_id", "week", "demand_pred", "horizon"])


def test_valid_frame_has_no_problem():
    assert xf.validate_forecast(good_frame(), ["A", "B"]) == []


@pytest.mark.parametrize("mutate,fragment", [
    (lambda d: d.drop(columns=["demand_pred"]), "missing mandatory"),
    (lambda d: d.assign(demand_pred=d["demand_pred"].where(d.index != 0, -1.0)), "negative"),
    (lambda d: d.assign(demand_pred=d["demand_pred"].where(d.index != 0, np.nan)), "missing values"),
    (lambda d: d.assign(week=d["week"].where(d.index != 0, "2017-08-22")), "Monday"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicated"),
    (lambda d: d.iloc[:-1], "exactly 4"),
    (lambda d: d.assign(week=d["week"].where(d.index != 1, "2017-09-25")), "consecutive"),
])
def test_contract_violations_are_detected(mutate, fragment):
    problems = xf.validate_forecast(mutate(good_frame()), ["A", "B"])
    assert any(fragment in p for p in problems), problems


def test_product_set_is_checked():
    assert any("missing from the file" in p for p in xf.validate_forecast(good_frame(("A",)), ["A", "B"]))
    assert any("unknown products" in p for p in xf.validate_forecast(good_frame(("A", "B", "Z")), ["A", "B"]))


def test_export_end_to_end(tmp_path):
    pytest.importorskip("keras")
    import test_predict as tp
    from src.forecast import data_loader as dl
    from src.forecast import ensemble as en
    from src.forecast import sequence_generator as sg

    base = dl.load_config()
    cfg = {**base, "forecast": {"window_size": 12, "horizon": 4, "target_mode": "delta", "season_length": 52},
           "split": {"val_weeks": 26, "test_weeks": 26},
           "model": {**base["model"], "epochs": 2, "batch_size": 16, "lstm_units": [8, 4], "dense_units": 8,
                     "use_seasonal_input": True}}
    weekly = tp.synthetic_weekly()
    (tmp_path / "outputs").mkdir()
    weekly.to_csv(tmp_path / "outputs" / "sales_weekly.csv", index=False)
    sg.prepare_dataset(cfg, weekly=weekly, holidays=tp.synthetic_holidays(), out_root=tmp_path, save=True)
    en.run_ensemble(cfg, seeds=[1, 2], root=tmp_path, weekly=weekly)

    res = xf.export_forecast(cfg, tmp_path)
    df = pd.read_csv(tmp_path / "outputs" / "demand_forecast.csv")
    assert list(df.columns) == ["product_id", "week", "demand_pred", "horizon"]
    assert len(df) == 12 and xf.validate_forecast(df, ["A", "B", "C"]) == []
    assert (tmp_path / "outputs" / "demand_forecast_details.csv").exists()
    assert (tmp_path / "outputs" / "plots" / "08_forecast_next_weeks.png").exists()
    assert res["info"]["n_products"] == 3
