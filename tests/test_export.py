from pathlib import Path

import onnx
import pandas as pd

from prodml.export import export_to_onnx
from prodml.features.build_features import get_preprocessor
from prodml.models.train import train_model


def test_export_to_onnx(
    tmp_path: Path,
):
    x_train = pd.DataFrame(
        {
            "PU_DO": [
                "1_2",
                "1_3",
                "2_3",
                "2_4",
            ],
            "trip_distance": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        }
    )

    y_train = pd.Series(
        [
            10.0,
            12.0,
            16.0,
            20.0,
        ]
    )

    preprocessor = get_preprocessor()

    model = train_model(
        X_train=x_train,
        y_train=y_train,
        preprocessor=preprocessor,
    )

    output_path = (
        tmp_path / "taxi_duration.onnx"
    )

    export_to_onnx(
        model,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    exported_model = onnx.load(
        str(output_path)
    )

    onnx.checker.check_model(
        exported_model
    )

    input_names = {
        input_tensor.name
        for input_tensor in exported_model.graph.input
    }

    assert "PU_DO" in input_names
    assert "trip_distance" in input_names