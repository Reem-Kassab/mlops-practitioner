from pydantic import BaseModel, Field, RootModel


class PredictionRequest(BaseModel):
    """Request payload for a single prediction."""

    PULocationID: int = Field(
        ...,
        description="Pickup location ID.",
        examples=[138],
    )

    DOLocationID: int = Field(
        ...,
        description="Dropoff location ID.",
        examples=[161],
    )

    trip_distance: float = Field(
        ...,
        gt=0,
        lt=200,
        description="Trip distance in miles.",
        examples=[2.5],
    )


class PredictionResponse(BaseModel):
    """Response payload for a single prediction."""

    prediction: float = Field(
        ...,
        description="Predicted trip duration in minutes.",
        examples=[18.42],
    )

    model_version: str = Field(
        ...,
        description="Version of the model serving the prediction.",
        examples=["v0.1.0"],
    )

    correlation_id: str = Field(
        ...,
        description="Unique identifier for tracing the request.",
        examples=["0d6a1f8d-8f3d-4a9a-9c0e-5f8b9e8dbb31"],
    )

    latency_ms: float = Field(
        ...,
        description="Prediction execution latency in milliseconds.",
        examples=[2.31],
    )


class BatchPredictionRequest(
    RootModel[list[PredictionRequest]]
):
    """Request payload for batch prediction."""

    root: list[PredictionRequest] = Field(
        min_length=1,
        max_length=1000,
        description="List of prediction requests.",
    )


class BatchPredictionResponse(
    RootModel[list[PredictionResponse]]
):
    """Response payload for batch prediction."""

    root: list[PredictionResponse]