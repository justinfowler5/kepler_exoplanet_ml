"""API endpoint contract tests."""

from __future__ import annotations

from unittest.mock import patch


def test_run_experiment_returns_202(app_client) -> None:
    with patch(
        "kepler_engine.api.v1.experiments.enqueue_experiment",
        return_value="11111111-1111-1111-1111-111111111111",
    ) as enq:
        response = app_client.post(
            "/api/v1/experiment/run",
            json={"model_type": "random_forest", "test_size": 0.2, "cv_folds": 2},
        )
    assert response.status_code == 202
    body = response.json()
    assert body["run_id"] == "11111111-1111-1111-1111-111111111111"
    assert body["status"] == "PENDING"
    enq.assert_called_once()


def test_get_unknown_experiment_404(app_client) -> None:
    response = app_client.get("/api/v1/experiment/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")


def test_predict_validation_failure(app_client) -> None:
    response = app_client.post("/api/v1/predict", json={"records": [{"koi_period": -1}]})
    assert response.status_code == 422


def test_predict_requires_champion(app_client) -> None:
    transit = {
        "koi_period": 9.488,
        "koi_time0bk": 170.539,
        "koi_impact": 0.146,
        "koi_duration": 2.958,
        "koi_depth": 615.8,
        "koi_prad": 2.26,
        "koi_teq": 793.0,
        "koi_insol": 93.59,
        "koi_model_snr": 35.8,
        "koi_tce_plnt_num": 1,
        "koi_steff": 5455.0,
        "koi_slogg": 4.467,
        "koi_srad": 0.927,
        "koi_kepmag": 15.347,
    }
    response = app_client.post("/api/v1/predict", json={"records": [transit]})
    assert response.status_code == 404
