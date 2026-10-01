import logging
from time import perf_counter

from fastapi import APIRouter, Depends, Request

from prodml.api.dependencies import get_predictor
from prodml.api.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.models.predictor import DurationPredictor


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["prediction"],
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict one taxi trip duration",
)
def predict(
    payload: PredictionRequest,
    request: Request,
    predictor: DurationPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Generate a prediction for a single taxi trip."""

    start = perf_counter()

    features = {
        "PULocationID": payload.PULocationID,
        "DOLocationID": payload.DOLocationID,
        "trip_distance": payload.trip_distance,
    }

    prediction = predictor.predict_one(features)

    latency_ms = (perf_counter() - start) * 1000

    correlation_id = request.state.correlation_id
    model_version = request.app.state.model_version

    logger.info(
        "Prediction served",
        extra={
            "correlation_id": correlation_id,
            "latency_ms": round(latency_ms, 3),
        },
    )

    return PredictionResponse(
        prediction=prediction,
        model_version=model_version,
        correlation_id=correlation_id,
        latency_ms=round(latency_ms, 3),
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict multiple taxi trip durations",
)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
    predictor: DurationPredictor = Depends(get_predictor),
) -> BatchPredictionResponse:
    """Generate predictions for a batch of taxi trips."""

    start = perf_counter()

    features = [
        {
            "PULocationID": item.PULocationID,
            "DOLocationID": item.DOLocationID,
            "trip_distance": item.trip_distance,
        }
        for item in payload.root
    ]

    predictions = predictor.predict_batch(features)

    latency_ms = (perf_counter() - start) * 1000

    correlation_id = request.state.correlation_id
    model_version = request.app.state.model_version

    logger.info(
        "Batch prediction served",
        extra={
            "correlation_id": correlation_id,
            "latency_ms": round(latency_ms, 3),
            "batch_size": len(features),
        },
    )

    responses = [
        PredictionResponse(
            prediction=prediction,
            model_version=model_version,
            correlation_id=correlation_id,
            latency_ms=round(latency_ms, 3),
        )
        for prediction in predictions
    ]

    return BatchPredictionResponse(root=responses)