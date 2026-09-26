import joblib
import pandas as pd
from typing import Dict, Any
import numpy as np

from taxi_duration.config import settings

class TaxiDurationPredictor:
    def __init__(self):
        self.model = None

    def load(self, model_path: str = settings.model_save_path):
        """Loads the trained pipeline."""
        self.model = joblib.load(model_path)
        return self

    def predict_batch(self, data: pd.DataFrame) -> np.ndarray:
        """Makes predictions on a batch of data."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
            
        if "PU_DO" not in data.columns and "PULocationID" in data.columns and "DOLocationID" in data.columns:
            data = data.copy()
            data["PU_DO"] = data["PULocationID"].astype(str) + "_" + data["DOLocationID"].astype(str)
            
        return self.model.predict(data)

    def predict_one(self, trip_data: Dict[str, Any]) -> float:
        """Makes a prediction for a single trip."""
        df = pd.DataFrame([trip_data])
        prediction = self.predict_batch(df)
        return float(prediction[0])

if __name__ == "__main__":
    predictor = TaxiDurationPredictor().load()
    sample_trip = {"PULocationID": 10, "DOLocationID": 50, "trip_distance": 3.5}
    pred = predictor.predict_one(sample_trip)
    print(f"Prediction: {pred:.2f} minutes")