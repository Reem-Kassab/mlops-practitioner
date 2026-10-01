from pathlib import Path

import numpy as np
import pandas as pd

from prodml.models.onnx_model import ONNXModel
from prodml.models.sklearn_model import SklearnModel
from prodml.utils.io import get_model_artifact_path,get_onnx_artifact_path,get_validation_sample_path


def test_pickle_and_onnx_predictions_match():
    validation_path = get_validation_sample_path()

    assert validation_path.exists(), (
        f"Validation sample not found: {validation_path}. "
        "Run the training pipeline first."
    )

    validation_data = pd.read_parquet(validation_path)

    assert len(validation_data) == 500

    features = validation_data.to_dict(orient="records")

    sklearn_model = SklearnModel(
        get_model_artifact_path()
    )
    sklearn_model.load()

    onnx_model = ONNXModel(
        get_onnx_artifact_path()
    )
    onnx_model.load()

    pickle_predictions = np.asarray(
        sklearn_model.predict_batch(features)
    )

    onnx_predictions = np.asarray(
        onnx_model.predict_batch(features)
    )

    np.testing.assert_allclose(
        pickle_predictions,
        onnx_predictions,
        atol=1e-4,
    )