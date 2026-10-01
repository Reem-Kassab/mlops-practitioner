from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health status of the inference service."""

    status: str = Field(
        ...,
        examples=["ok"],
    )

    model_loaded: bool = Field(
        ...,
        examples=[True],
    )


class MetadataResponse(BaseModel):
    """Metadata describing the currently served model."""

    model_version: str = Field(
        ...,
        examples=["v0.1.0"],
    )

    training_date: str = Field(
        ...,
        examples=["2026-09-30"],
    )

    feature_names: list[str] = Field(
        ...,
        examples=[["PU_DO", "trip_distance"]],
    )

    framework: str = Field(
        ...,
        examples=["ONNX Runtime"],
    )

    artifact_sha256: str = Field(
        ...,
        examples=[
            "7e5f4a1c4c3f2a5a5bd3c7d9f0f0b3c1"
        ],
    )