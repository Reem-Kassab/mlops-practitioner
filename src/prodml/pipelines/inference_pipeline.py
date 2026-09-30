from pathlib import Path
from prodml.models.predictor import DurationPredictor
from prodml.models.sklearn_model import SklearnModel
from prodml.utils.io import get_model_artifact_path


def create_predictor(model_path: Path | None = None) -> DurationPredictor:
    """Create and load the duration predictor."""

    artifact_path = model_path or get_model_artifact_path()

    model = SklearnModel(artifact_path)

    predictor = DurationPredictor(model)

    predictor.load()

    return predictor