from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import onnxruntime as ort

from prodml.models.base import ModelBase


class ONNXModel(ModelBase):
    """ONNX Runtime implementation of the model interface."""

    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self.session: ort.InferenceSession | None = None

    def load(self) -> None:
        """Load the ONNX model into an inference session."""
        if not self.artifact_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {self.artifact_path}"
            )

        self.session = ort.InferenceSession(
            str(self.artifact_path),
            providers=["CPUExecutionProvider"],
        )

    def predict_one(self, features: dict[str, Any]) -> float:
        """Generate one prediction."""
        if self.session is None:
            raise RuntimeError("ONNX model is not loaded.")

        features_df = pd.DataFrame([features])

        inputs = {
            "PU_DO": features_df[["PU_DO"]].to_numpy(),
            "trip_distance": features_df[
                ["trip_distance"]
            ].to_numpy(dtype=np.float32),
        }

        outputs = self.session.run(
            None,
            inputs,
        )

        return float(
            np.asarray(outputs[0]).reshape(-1)[0]
        )

    def predict_batch(self,features: list[dict[str, Any]]) -> list[float]:
        """Generate predictions for multiple observations."""
        if self.session is None:
            raise RuntimeError("ONNX model is not loaded.")

        features_df = pd.DataFrame(features)

        inputs = {
            "PU_DO": features_df[["PU_DO"]].to_numpy(),
            "trip_distance": features_df[
                ["trip_distance"]
            ].to_numpy(dtype=np.float32),
        }

        outputs = self.session.run(
            None,
            inputs,
        )

        return (
            np.asarray(outputs[0])
            .reshape(-1)
            .astype(float)
            .tolist()
        )