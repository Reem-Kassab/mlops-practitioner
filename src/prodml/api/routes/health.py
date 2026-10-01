from fastapi import APIRouter, HTTPException, Request, status

from prodml.api.schemas.health import HealthResponse


router = APIRouter(
    prefix="",
    tags=["health"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check service health",
)
def health(request: Request) -> HealthResponse:
    """Return healthy only when the model is loaded."""

    model_loaded = getattr(
        request.app.state,
        "model_loaded",
        False,
    )

    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
    )