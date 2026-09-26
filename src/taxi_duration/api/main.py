from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from taxi_duration.api.schemas import (
    BatchPredictionOutput,
    BatchTripInput,
    HealthResponse,
    PredictionOutput,
    TripInput,
)
from taxi_duration.logging_conf import setup_logging
from taxi_duration.predict import TaxiDurationPredictor

logger = setup_logging("taxi_duration_api")

predictor = TaxiDurationPredictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: managing the server.
    """
    logger.info("Starting up FastAPI server...")
    try:
        predictor.load()
        logger.info("Model loaded successfully into memory.")
    except (RuntimeError, OSError) as e:
        logger.error(f" Failed to load model: {e}")
    
    yield 
    
    logger.info("Shutting down FastAPI server...")

app = FastAPI(
    title="Taxi Duration Prediction API",
    description="MLOps API for predicting NYC taxi trip durations.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Checks if the API is running and the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor.model is not None
    )

@app.get("/metadata")
def get_metadata():
    """Returns metadata about the deployed model."""
    return {
        "model_name": "NYC Taxi Duration Baseline",
        "model_type": "Linear Regression Pipeline",
        "expected_features": ["PULocationID", "DOLocationID", "trip_distance"]
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(trip: TripInput):
    """Predicts the duration for a single trip."""
    logger.info(f"Received single prediction request for trip_distance={trip.trip_distance}")
    
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        pred = predictor.predict_one(trip.model_dump())
        return PredictionOutput(duration_minutes=pred)
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchTripInput):
    """Predicts durations for a batch of trips."""
    num_trips = len(batch.trips)
    logger.info(f"Received batch prediction request for {num_trips} trips")
    
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        trip_dicts = [trip.model_dump() for trip in batch.trips]
        df = pd.DataFrame(trip_dicts)
        
        preds = predictor.predict_batch(df)
        
        return BatchPredictionOutput(predictions=preds.tolist())
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))