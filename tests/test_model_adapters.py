from pathlib import Path

import joblib
import pandas as pd
import pytest

from prodml.export import export_to_onnx
from prodml.models.onnx_model import ONNXModel
from prodml.models.sklearn_model import SklearnModel


def _sample_model_features() -> list[dict]:
    return [
        {
            "PU_DO": "1_2",
            "trip_distance": 1.5,
        },
        {
            "PU_DO": "2_3",
            "trip_distance": 2.5,
        },
    ]


def test_sklearn_model_load_and_predict(
    trained_model,
    tmp_path: Path,
):
    artifact_path = (
        tmp_path / "taxi_duration.pkl"
    )

    joblib.dump(
        trained_model,
        artifact_path,
    )

    model = SklearnModel(
        artifact_path
    )

    model.load()

    assert model.model is not None

    features = _sample_model_features()

    prediction = model.predict_one(
        features[0]
    )

    predictions = model.predict_batch(
        features
    )

    assert isinstance(
        prediction,
        float,
    )

    assert isinstance(
        predictions,
        list,
    )

    assert len(predictions) == 2

    assert all(
        isinstance(value, float)
        for value in predictions
    )


def test_sklearn_model_requires_load(
    tmp_path: Path,
):
    artifact_path = (
        tmp_path / "missing.pkl"
    )

    model = SklearnModel(
        artifact_path
    )

    with pytest.raises(RuntimeError):
        model.predict_one(
            {
                "PU_DO": "1_2",
                "trip_distance": 1.5,
            }
        )

    with pytest.raises(RuntimeError):
        model.predict_batch(
            [
                {
                    "PU_DO": "1_2",
                    "trip_distance": 1.5,
                }
            ]
        )


def test_sklearn_model_load_failure(
    tmp_path: Path,
):
    artifact_path = (
        tmp_path / "invalid.pkl"
    )

    artifact_path.write_text(
        "not a pickle file"
    )

    model = SklearnModel(
        artifact_path
    )

    with pytest.raises(Exception):
        model.load()


def test_onnx_model_load_and_predict(
    trained_model,
    tmp_path: Path,
):
    onnx_path = (
        tmp_path / "taxi_duration.onnx"
    )

    export_to_onnx(
        trained_model,
        onnx_path,
    )

    model = ONNXModel(
        onnx_path
    )

    model.load()

    assert model.session is not None

    features = _sample_model_features()

    prediction = model.predict_one(
        features[0]
    )

    predictions = model.predict_batch(
        features
    )

    assert isinstance(
        prediction,
        float,
    )

    assert isinstance(
        predictions,
        list,
    )

    assert len(predictions) == 2

    assert all(
        isinstance(value, float)
        for value in predictions
    )


def test_onnx_model_requires_load(
    tmp_path: Path,
):
    onnx_path = (
        tmp_path / "model.onnx"
    )

    model = ONNXModel(
        onnx_path
    )

    with pytest.raises(RuntimeError):
        model.predict_one(
            {
                "PU_DO": "1_2",
                "trip_distance": 1.5,
            }
        )

    with pytest.raises(RuntimeError):
        model.predict_batch(
            [
                {
                    "PU_DO": "1_2",
                    "trip_distance": 1.5,
                }
            ]
        )