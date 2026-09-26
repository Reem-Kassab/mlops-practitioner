import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from taxi_duration.api.main import app, predictor

@pytest.fixture
def client():
    predictor.load()
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """Testing the /health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

def test_metadata(client):
    """Testing the /metadata endpoint"""
    response = client.get("/metadata")
    assert response.status_code == 200
    assert "model_type" in response.json()

def test_predict_single_trip(client):
    """Testing valid single prediction"""
    payload = {
        "PULocationID": 236,
        "DOLocationID": 239,
        "trip_distance": 2.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "duration_minutes" in response.json()
    assert isinstance(response.json()["duration_minutes"], float)

def test_predict_batch_trips(client):
    """Testing batch predictions"""
    payload = {
        "trips": [
            {"PULocationID": 236, "DOLocationID": 239, "trip_distance": 2.5},
            {"PULocationID": 10, "DOLocationID": 50, "trip_distance": 0.8}
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 2

def test_invalid_trip_distance(client):
    """Testing Pydantic validation (Negative distance)"""
    payload = {
        "PULocationID": 10,
        "DOLocationID": 50,
        "trip_distance": -3.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422