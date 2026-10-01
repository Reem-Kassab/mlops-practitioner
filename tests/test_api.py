from unittest.mock import Mock

from prodml.api.schemas.prediction import (
    BatchPredictionResponse,
    PredictionResponse,
)


def test_health_returns_200(client):
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_happy_path(
    client,
    sample_features,
):
    response = client.post(
        "/predict",
        json=sample_features,
    )

    assert response.status_code == 200

    body = response.json()

    PredictionResponse.model_validate(body)

    assert body["prediction"] == 12.5
    assert body["model_version"] == "dev"
    assert "correlation_id" in body
    assert "latency_ms" in body


def test_predict_invalid_payload_returns_422(
    client,
    sample_features,
):
    invalid_payload = {
        **sample_features,
        "trip_distance": -5,
    }

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422

    body = response.json()

    assert "detail" in body
    assert "correlation_id" in body


def test_predict_response_schema_matches(
    client,
    sample_features,
):
    response = client.post(
        "/predict",
        json=sample_features,
    )

    assert response.status_code == 200

    validated = PredictionResponse.model_validate(
        response.json()
    )

    assert isinstance(
        validated.prediction,
        float,
    )

    assert isinstance(
        validated.model_version,
        str,
    )

    assert isinstance(
        validated.correlation_id,
        str,
    )

    assert isinstance(
        validated.latency_ms,
        float,
    )


def test_batch_prediction(client):
    payload = [
        {
            "PULocationID": 138,
            "DOLocationID": 161,
            "trip_distance": 2.5,
        },
        {
            "PULocationID": 100,
            "DOLocationID": 186,
            "trip_distance": 4.1,
        },
    ]

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 200

    validated = (
        BatchPredictionResponse.model_validate(
            response.json()
        )
    )

    assert len(validated.root) == 2


def test_metadata(client):
    response = client.get("/metadata")

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "dev"
    assert body["feature_names"] == [
        "PU_DO",
        "trip_distance",
    ]

    assert "training_date" in body
    assert "framework" in body
    assert "artifact_sha256" in body


def test_internal_error_returns_500_without_traceback(
    client,
    sample_features,
):
    predictor = client.app.state.predictor

    predictor.predict_one = Mock(
        side_effect=RuntimeError(
            "secret internal failure"
        )
    )

    response = client.post(
        "/predict",
        json=sample_features,
    )

    assert response.status_code == 500

    body = response.json()

    assert body["detail"] == (
        "Internal server error."
    )

    assert "correlation_id" in body

    assert "secret internal failure" not in response.text