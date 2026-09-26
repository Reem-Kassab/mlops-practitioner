from typing import Any
import numpy as np
import pandas as pd
import onnxruntime as ort
from taxi_duration.config import settings
from taxi_duration.features import build_features
import logging

logger = logging.getLogger(__name__)

class TaxiDurationPredictor:
    def __init__(self):
        self.session = None

    def load(self, model_path: str = None):
        """Loads the ONNX optimized model."""
        if model_path is None:
            model_path = str(settings.model_save_path).replace(".pkl", ".onnx")
        else:
            model_path = str(model_path).replace(".pkl", ".onnx")
            
        logger.info(f"Loading ONNX model from {model_path}")
        self.session = ort.InferenceSession(model_path)
        return self

    def _prepare_inputs(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Prepares inputs matching the ONNX model expectations using the feature engineering function."""
        if "duration" not in df.columns:
            df["duration"] = 0.0
        df = build_features(df)

        onnx_inputs = {
            'PU_DO': df['PU_DO'].to_numpy().reshape(-1, 1).astype(str),
            'trip_distance': df['trip_distance'].to_numpy().reshape(-1, 1).astype(np.float32)
        }
        return onnx_inputs

    def predict_one(self, features: dict[str, Any]) -> float:
        """Runs inference for a single trip using ONNX."""
        if self.session is None:
            self.load()

        df = pd.DataFrame([features])
        onnx_inputs = self._prepare_inputs(df)
        result = self.session.run(None, onnx_inputs)
        preds = result[0].flatten()
        return float(preds[0])


    def predict_batch(self, trips: list[dict[str, Any]]) -> np.ndarray:
        """Runs inference for a batch of trips using ONNX."""
        if self.session is None:
            self.load()

        df = pd.DataFrame(trips)
        onnx_inputs = self._prepare_inputs(df)
        
        result = self.session.run(None, onnx_inputs)
        return result[0].flatten()