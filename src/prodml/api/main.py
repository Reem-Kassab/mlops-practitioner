import hashlib
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from prodml.api.routes.health import router as health_router
from prodml.api.routes.metadata import router as metadata_router
from prodml.api.routes.predict import router as predict_router
from prodml.pipelines.inference_pipeline import create_predictor
from prodml.logging_conf import configure_logging


logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    "PU_DO",
    "trip_distance",
]


def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 for a model artifact."""

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_artifact_path(predictor) -> Path | None:
    """Try to retrieve the artifact path from the loaded model."""

    model = getattr(predictor, "model", None)

    artifact_path = getattr(
        model,
        "artifact_path",
        None,
    )

    if artifact_path is None:
        return None

    return Path(artifact_path)


def get_framework(predictor) -> str:
    """Return a human-readable serving framework name."""

    model = getattr(predictor, "model", None)

    model_name = type(model).__name__

    if model_name == "ONNXModel":
        return "ONNX Runtime"

    if model_name == "SklearnModel":
        return "scikit-learn / joblib"

    return model_name


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the predictor once when the application starts."""

    configure_logging()

    logger.info("Starting inference service")

    try:
        predictor = create_predictor()

        app.state.predictor = predictor
        app.state.model_loaded = True

        app.state.model_version = os.getenv(
            "MODEL_VERSION",
            "dev",
        )

        app.state.training_date = os.getenv(
            "MODEL_TRAINING_DATE",
            "unknown",
        )

        app.state.feature_names = FEATURE_NAMES

        app.state.framework = get_framework(
            predictor
        )

        artifact_path = get_artifact_path(
            predictor
        )

        if artifact_path is not None and artifact_path.exists():
            app.state.artifact_sha256 = (
                calculate_sha256(artifact_path)
            )
        else:
            app.state.artifact_sha256 = "unknown"

        logger.info(
            "Inference service started",
            extra={
                "model_version": app.state.model_version,
                "framework": app.state.framework,
            },
        )

    except Exception:
        app.state.model_loaded = False

        logger.exception(
            "Failed to load inference model"
        )

        raise

    yield

    logger.info("Shutting down inference service")


app = FastAPI(
    title="Taxi Duration Prediction API",
    description=(
        "Production inference API for NYC taxi "
        "trip-duration prediction."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(
    request: Request,
    call_next,
):
    """Generate a correlation ID for every request."""

    correlation_id = request.headers.get(
        "X-Request-ID"
    )

    if not correlation_id:
        correlation_id = str(uuid4())

    request.state.correlation_id = correlation_id

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled request exception",
            extra={
                "correlation_id": correlation_id,
            },
        )
        raise

    response.headers["X-Request-ID"] = correlation_id

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Return a clean 422 validation response."""

    correlation_id = getattr(
        request.state,
        "correlation_id",
        "unknown",
    )

    logger.warning(
        "Request validation failed",
        extra={
            "correlation_id": correlation_id,
        },
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": jsonable_encoder(
                exc.errors()
            ),
            "correlation_id": correlation_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    """Return a safe 500 response without exposing internals."""

    correlation_id = getattr(
        request.state,
        "correlation_id",
        "unknown",
    )

    logger.exception(
        "Unhandled application exception",
        extra={
            "correlation_id": correlation_id,
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
            "correlation_id": correlation_id,
        },
    )


app.include_router(health_router)
app.include_router(metadata_router)
app.include_router(predict_router)