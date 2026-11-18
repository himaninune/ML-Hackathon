import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from src.api.app import app, scorer


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert {"status", "model_loaded", "road_network_loaded"} <= set(response.json())


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


def test_invalid_prediction_request(client):
    assert client.post("/api/predict", json={}).status_code == 422
    assert client.post("/api/predict", json={"trip_id": "../secret"}).status_code == 422


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
