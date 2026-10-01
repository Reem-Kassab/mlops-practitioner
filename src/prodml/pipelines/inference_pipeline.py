from pathlib import Path
from prodml.models.predictor import DurationPredictor
from prodml.models.onnx_model import ONNXModel
from prodml.utils.io import get_model_artifact_path


def create_predictor(model_path: Path | None = None) -> DurationPredictor:
    """Create and load the duration predictor."""

    artifact_path = model_path or get_model_artifact_path()

    onnx_artifact_path = (artifact_path.with_suffix(".onnx"))

    model = ONNXModel(onnx_artifact_path)

    predictor = DurationPredictor(model)

    predictor.load()

    return predictor