from pathlib import Path

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.pipeline import Pipeline


def export_to_onnx(model: Pipeline, output_path: Path,) -> None:
    """Export a fitted scikit-learn pipeline to ONNX."""

    initial_types = [
        ("PU_DO", StringTensorType([None, 1])),
        ("trip_distance", FloatTensorType([None, 1])),
    ]

    model_onnx = convert_sklearn(
        model,
        initial_types=initial_types,
    )

    onnx.checker.check_model(model_onnx)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("wb") as file:
        file.write(model_onnx.SerializeToString())