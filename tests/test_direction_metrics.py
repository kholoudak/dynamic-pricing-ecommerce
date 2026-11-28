"""Tests for the direction (classification-style) metrics (run: pytest tests -v)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecast import direction_metrics as dm  # noqa: E402


def test_confusion_metrics_hand_example():
    # truth: 4 positives, 6 negatives ; predictions: TP=3, FN=1, FP=2, TN=4
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    p = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0], bool)
    r = dm.confusion_metrics(t, p)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (3, 1, 2, 4)
    assert r["accuracy"] == pytest.approx(0.7)
    assert r["precision"] == pytest.approx(3 / 5) and r["recall"] == pytest.approx(3 / 4)
    assert r["f1"] == pytest.approx(2 * 0.6 * 0.75 / 1.35)
    assert r["specificity"] == pytest.approx(4 / 6)
    assert r["type1_error"] == pytest.approx(1 - r["specificity"])
    assert r["type2_error"] == pytest.approx(1 - r["recall"])


def test_never_positive_prediction_has_undefined_precision():
    r = dm.confusion_metrics([1, 0, 1, 0], [0, 0, 0, 0])
    assert np.isnan(r["precision"]) and np.isnan(r["f1"]) and r["recall"] == 0 and r["specificity"] == 1


def test_size_mismatch_raises():
    with pytest.raises(ValueError):
        dm.confusion_metrics([1, 0], [1])


def test_compute_direction_and_naive_never_predicts_increase():
    weeks = pd.date_range("2017-01-02", periods=3, freq="7D")
    weekly = pd.DataFrame({"product_id": "A", "week": weeks, "sales": [100.0, 110.0, 90.0]})
    pred = pd.DataFrame({"product_id": "A", "origin_week": [weeks[0], weeks[1]], "horizon": [1, 1],
                         "actual": [110.0, 90.0], "lstm": [105.0, 95.0], "naive": [100.0, 110.0]})
    table = dm.add_last_observed(pred, weekly)
    assert table["last_obs"].tolist() == [100.0, 110.0]
    res = dm.compute_direction(table)
    assert res["lstm"]["tp"] == 1 and res["lstm"]["tn"] == 1 and res["lstm"]["accuracy"] == 1.0
    assert res["naive"]["tp"] == 0 and res["naive"]["fp"] == 0
    assert res["_share_of_real_increases"] == pytest.approx(0.5)


def test_missing_origin_week_raises():
    weekly = pd.DataFrame({"product_id": ["A"], "week": [pd.Timestamp("2017-01-02")], "sales": [1.0]})
    pred = pd.DataFrame({"product_id": ["A"], "origin_week": [pd.Timestamp("2018-01-01")], "horizon": [1],
                         "actual": [1.0], "lstm": [1.0]})
    with pytest.raises(ValueError):
        dm.add_last_observed(pred, weekly)
