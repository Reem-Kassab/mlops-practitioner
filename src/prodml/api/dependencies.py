from typing import cast

from fastapi import HTTPException, Request, status

from prodml.models.predictor import DurationPredictor


def get_predictor(request: Request) -> DurationPredictor:
    """Return the application-scoped predictor."""

    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready.",
        )

    return cast(DurationPredictor, predictor)