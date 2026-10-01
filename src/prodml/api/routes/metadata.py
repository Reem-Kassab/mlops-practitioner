from fastapi import APIRouter, Request

from prodml.api.schemas.health import MetadataResponse


router = APIRouter(
    prefix="",
    tags=["metadata"],
)


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Get model metadata",
)
def metadata(request: Request) -> MetadataResponse:
    """Return metadata for the currently served model."""

    return MetadataResponse(
        model_version=request.app.state.model_version,
        training_date=request.app.state.training_date,
        feature_names=request.app.state.feature_names,
        framework=request.app.state.framework,
        artifact_sha256=request.app.state.artifact_sha256,
    )