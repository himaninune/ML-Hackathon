import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from src.api.app import app, scorer
from src.services.metrics_service import metrics_service


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert {"status", "model_loaded", "road_network_loaded"} <= set(response.json())


def test_frontend_cors_origin_is_allowed(client):
    response = client.options("/api/health", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_trip_list_and_details_hide_routes(client):
    trips = client.get("/api/trips").json()["trips"]
    assert len(trips) == 10
    assert all("route" not in str(item).lower() for item in trips)
    details = client.get(f"/api/trips/{trips[0]['trip_id']}")
    assert details.status_code == 200
    assert "route" not in details.json()


def test_invalid_and_missing_trip(client):
    assert client.get("/api/trips/not-a-trip").status_code == 404
    assert client.get("/api/trips/99999999").status_code == 404


def test_experiment_metrics_endpoints_read_artifacts(client):
    xgb = client.get("/api/metrics/xgboost")
    assert xgb.status_code == 200
    xgb_body = xgb.json()
    assert xgb_body["available"] is True
    assert xgb_body["evaluation_stage"] == "candidate_classification"
    assert xgb_body["metrics"]["selected_model"] == "XGBoost (Untuned Baseline)"
    assert xgb_body["metrics"]["baseline_metrics"]["metrics"]["student_test_top1_accuracy"] == 0.8371
    assert xgb_body["metrics"]["bayesian_tuned_metrics"]["metrics"]["student_test_top1_accuracy"] == 0.8396

    viterbi = client.get("/api/metrics/viterbi")
    assert viterbi.status_code == 200
    viterbi_body = viterbi.json()
    assert viterbi_body["available"] is True
    assert viterbi_body["evaluation_stage"] == "route_reconstruction"
    assert viterbi_body["evaluation_split"] == "student_test"
    assert viterbi_body["metrics"]["viterbi_sequence_top1_accuracy"] == 0.8003
    assert viterbi_body["metrics"]["transition_log_priors"]["disconnected"] == -7.5

    overview = client.get("/api/metrics/overview")
    assert overview.status_code == 200
    assert overview.json()["xgboost"]["metrics"] == xgb_body["metrics"]
    assert overview.json()["viterbi"]["metrics"] == viterbi_body["metrics"]


def test_missing_and_invalid_metric_files_return_unavailable(client, tmp_path, monkeypatch):
    monkeypatch.setattr(metrics_service, "xgboost_path", tmp_path / "missing.json")
    missing = client.get("/api/metrics/xgboost")
    assert missing.status_code == 200
    assert missing.json()["available"] is False
    assert "missing" in missing.json()["reason"].lower()
    assert "tmp_path" not in missing.text

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{invalid json", encoding="utf-8")
    monkeypatch.setattr(metrics_service, "viterbi_path", invalid_path)
    invalid = client.get("/api/metrics/viterbi")
    assert invalid.status_code == 200
    assert invalid.json()["available"] is False
    assert "invalid json" in invalid.json()["reason"].lower()
    assert str(tmp_path) not in invalid.text


def test_invalid_prediction_request(client):
    assert client.post("/api/predict", json={}).status_code == 422
    assert client.post("/api/predict", json={"trip_id": "../secret"}).status_code == 422


def test_prediction_reports_missing_model(client, monkeypatch):
    monkeypatch.setattr(scorer, "model", None)
    response = client.post("/api/predict", json={"trip_id": "00000000"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Selected XGBoost model is unavailable"


def test_prediction_structure(client):
    if not scorer.loaded:
        pytest.skip("Selected XGBoost artifact is unavailable")
    trip_id = client.get("/api/trips").json()["trips"][0]["trip_id"]
    response = client.post("/api/predict", json={"trip_id": trip_id})
    assert response.status_code == 200, response.text
    body = response.json()
    assert {"trip_id", "status", "prediction", "evaluation", "processing_time_seconds"} <= set(body)
    assert {"matched_observations", "total_observations", "predicted_route", "predicted_coordinates"} <= set(body["prediction"])
    assert "ground_truth" not in str(body).lower()


def test_swagger_openapi(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
