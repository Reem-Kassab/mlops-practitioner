from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from prodml.models.base import ModelBase
import logging

logger = logging.getLogger(__name__)


class SklearnModel(ModelBase):
    """Scikit-learn implementation of the model interface."""

    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self.model = None

    def load(self) -> None:
        """Load the trained sklearn pipeline from disk."""
        try:
            self.model = joblib.load(self.artifact_path)
            logger.info("Model loaded successfully")
        except Exception:
            logger.error(
                "Model loading failed",
                exc_info=True,
            )
            raise

    def predict_one(self, features: dict[str, Any]) -> float:
        """Generate a prediction for one observation."""
        if self.model is None:
            raise RuntimeError(
                "the model does not load"
            )
        features_df=pd.DataFrame([features])
        prediction=self.model.predict(features_df)
        return float(prediction[0])

    def predict_batch(self,features: list[dict[str, Any]],) -> list[float]:
        """Generate predictions for multiple observations."""
        if self.model is None:
             raise RuntimeError(
                "the model does not load"
            )
        features_df=pd.DataFrame(features)
        predictions=self.model.predict(features_df)
        return predictions.tolist()
        