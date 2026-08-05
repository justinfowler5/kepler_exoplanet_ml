"""Health endpoint tests."""

from __future__ import annotations


def test_liveness_always_200(app_client) -> None:
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_checks(app_client) -> None:
    response = app_client.get("/health/ready")
    # Redis is fake (healthy); MLflow file store should ping OK.
    body = response.json()
    assert "checks" in body
    names = {c["name"] for c in body["checks"]}
    assert {"redis", "mlflow", "champion_model"} <= names
