"""Tests for the one-shot final evaluation (run: pytest tests -v)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("keras")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_predict as tp  # noqa: E402
from src.forecast import data_loader as dl  # noqa: E402
from src.forecast import ensemble as en  # noqa: E402
from src.forecast import final_evaluation as fe  # noqa: E402
from src.forecast import sequence_generator as sg  # noqa: E402


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("final")
    base = dl.load_config()
    cfg = {**base, "forecast": {"window_size": 12, "horizon": 4, "target_mode": "delta", "season_length": 52},
           "split": {"val_weeks": 26, "test_weeks": 26},
           "model": {**base["model"], "epochs": 2, "batch_size": 16, "lstm_units": [8, 4], "dense_units": 8,
                     "use_seasonal_input": True}}
    weekly = tp.synthetic_weekly()
    (root / "outputs").mkdir()
    weekly.to_csv(root / "outputs" / "sales_weekly.csv", index=False)
    sg.prepare_dataset(cfg, weekly=weekly, holidays=tp.synthetic_holidays(), out_root=root, save=True)
    en.run_ensemble(cfg, seeds=[1, 2], root=root, weekly=weekly)
    return cfg, root, weekly


def test_requires_final_flag(project):
    cfg, root, weekly = project
    with pytest.raises(ValueError):
        fe.run_final_evaluation(cfg, final=False, root=root, weekly=weekly)
    assert not (root / "outputs" / fe.RESULT_FILE).exists()


def test_runs_once_writes_reports_and_refuses_second_run(project):
    cfg, root, weekly = project
    res = fe.run_final_evaluation(cfg, final=True, root=root, weekly=weekly)
    assert set(res["overall"]) == {"hybrid", "lstm", "lstm_single", "naive", "moving_average", "seasonal_naive"}
    assert all(np.isfinite(v["WAPE"]) for v in res["overall"].values())
    for f in (fe.RESULT_FILE, "evaluation_test_final_by_product.csv", "predictions_test.csv", "metrics_summary.json",
              "plots/09_final_test_comparison.png", "plots/09_final_test_forecast_by_product.png"):
        assert (root / "outputs" / f).exists(), f
    with pytest.raises(ValueError):
        fe.run_final_evaluation(cfg, final=True, root=root, weekly=weekly)


def test_hybrid_is_the_weighted_mean(project):
    cfg, root, weekly = project
    prepared = sg.load_prepared(cfg, root=root)
    import json
    from src.forecast import model as md
    from src.forecast import train as tr
    em = json.loads((root / "models" / "ensemble_meta.json").read_text())
    members = [{"model": md.load_model(root / "models" / f), "model_meta": tr.load_model_meta(root / "models" / m)}
               for f, m in zip(em["files"], em["meta_files"])]
    res = fe.compute_final(cfg, prepared, members, weekly, 0.5)
    t = res["predictions"]
    assert np.allclose(t["hybrid"], np.clip(0.5 * t["lstm"] + 0.5 * t["seasonal_naive"], 0, None))
