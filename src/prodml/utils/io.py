from prodml.utils.config import PROJECT_ROOT, settings
from pathlib import Path

def get_raw_data_path() -> str:
    """getting the path of the data"""
    data_path = (
        PROJECT_ROOT
        / settings.data.raw_path
        / "green_tripdata_2026-01.parquet"
     )
    return data_path

def get_model_artifact_path()->str:
    """gitting the path of the model"""
    model_path = (
        PROJECT_ROOT
        / settings.model.artifact_dir
        / settings.model.artifact_name
    )
    return model_path


def get_onnx_artifact_path() -> Path:
    """Return the path to the ONNX model artifact."""
    return (
        PROJECT_ROOT
        / settings.model.artifact_dir
        / settings.model.onnx_artifact_name
    )

def get_validation_sample_path() -> Path:
    """Return the path to the serialization validation sample."""
    return (
        PROJECT_ROOT
        / settings.data.processed_path
        / settings.data.validation_sample_name
    )