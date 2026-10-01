from typing import Any

import pandas as pd

from prodml.features.build_features import build_feature
from prodml.models.base import ModelBase
import logging

logger = logging.getLogger(__name__)

class DurationPredictor:
    def __init__(self, model: ModelBase):
        self.model = model

    def load(self) -> None:
        """Delegate loading to the underlying model."""
        self.model.load()

    def predict_one(self, feature: dict[str, Any]) -> float:
        """Generate one taxi-duration prediction."""
        pu = feature["PULocationID"]
        do = feature["DOLocationID"]
        distance = feature["trip_distance"]

        pu_do = build_feature(
            pd.Series([pu]),
            pd.Series([do]),
        ).iloc[0]

        model_features = {
            "PU_DO": pu_do,
            "trip_distance": distance,
        }
        logger.debug(
            "Prepared feature vector: %s",
            model_features,
        )
        return self.model.predict_one(model_features)

    def predict_batch(
        self,
        features: list[dict[str, Any]],
    ) -> list[float]:
        """Generate predictions for multiple taxi trips."""
        pu = pd.Series(
            [feature["PULocationID"] for feature in features]
        )
        do = pd.Series(
            [feature["DOLocationID"] for feature in features]
        )

        pu_do = build_feature(pu, do)

        model_features = [
            {
                "PU_DO": pu_do.iloc[index],
                "trip_distance": features[index]["trip_distance"],
            }
            for index in range(len(features))
        ]

        logger.debug(
            "Prepared feature vector: %s",
            model_features,
        )
        return self.model.predict_batch(model_features)