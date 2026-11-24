"""Tests for src/pipeline/api.py (Agent 3, Phase 7). Uses the mock data fixture from conftest.py."""
import pytest
from fastapi.testclient import TestClient

import src.pipeline.api as api_module
from src.pipeline.api import _clean, create_app
from src.pipeline.pipeline import PipelineError


@pytest.fixture
def client(mock_cfg):
    return TestClient(create_app(mock_cfg))


@pytest.fixture
def ready_client(client):
    assert client.post("/refresh").status_code == 200
    return client


def test_price_before_any_refresh_is_503(client):
    r = client.get("/price/1")
    assert r.status_code == 503 and "POST /refresh" in r.json()["detail"]


def test_refresh_returns_summary(client):
    r = client.post("/refresh", json={"run_regression": False})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["n_products"] == 9 and body["n_rows"] == 36
    assert "ASSUMED elasticity" in body["note"]


def test_refresh_works_without_a_body(client):
    assert client.post("/refresh").status_code == 200


def test_get_price_has_all_the_required_information(ready_client):
    r = ready_client.get("/price/1")
    assert r.status_code == 200
    b = r.json()
    for key in ["price_current", "price_recommended", "demand_current", "demand_recommended",
                "profit_current", "profit_recommended", "estimated_margin_gain", "reasoning",
                "reputation_score", "schedule", "simulated", "note"]:
        assert key in b
    assert b["product_id"] == "1" and len(b["schedule"]) == 4 and b["simulated"] is True
    assert 0.8 * b["price_current"] <= b["price_recommended"] <= 1.2 * b["price_current"]   # constraints respected
    assert len(b["reasoning"]) > 20


def test_get_price_for_a_specific_week(ready_client):
    first = ready_client.get("/price/1").json()
    weeks = [w["week"] for w in first["schedule"]]
    r = ready_client.get("/price/1", params={"week": weeks[2]})
    assert r.status_code == 200 and r.json()["week"] == weeks[2]


def test_unknown_week_is_404_and_lists_available_weeks(ready_client):
    r = ready_client.get("/price/1", params={"week": "2030-01-01"})
    assert r.status_code == 404 and "Available weeks" in r.json()["detail"]


def test_unknown_product_is_404(ready_client):
    r = ready_client.get("/price/99999")
    assert r.status_code == 404 and "not found" in r.json()["detail"]


def test_invalid_inputs_are_422(ready_client):
    assert ready_client.get("/price/bad$id").status_code == 422             # forbidden character
    assert ready_client.get("/price/1", params={"week": "not-a-date"}).status_code == 422
    assert ready_client.post("/refresh", json={"rebuild_mapping": "maybe"}).status_code == 422


def test_failed_refresh_keeps_serving_the_previous_results(ready_client, monkeypatch):
    def boom(*args, **kwargs):
        raise PipelineError("Constraint audit failed: test")
    monkeypatch.setattr(api_module, "run_pipeline", boom)
    r = ready_client.post("/refresh")
    assert r.status_code == 500 and "Refresh failed" in r.json()["detail"]
    assert ready_client.get("/price/1").status_code == 200                  # old data still available


def test_concurrent_refresh_is_rejected_with_409(mock_cfg):
    app = create_app(mock_cfg)
    client = TestClient(app)
    app.state.lock.acquire()                                                 # pretend a refresh is running
    try:
        assert client.post("/refresh").status_code == 409
    finally:
        app.state.lock.release()


def test_clean_converts_nan_to_none():
    assert _clean(float("nan")) is None and _clean(None) is None and _clean(2.5) == 2.5


def test_openapi_documentation_lists_both_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/price/{product_id}" in paths and "/refresh" in paths
    assert client.get("/docs").status_code == 200
