from prodml.features.build_features import build_feature
from prodml.models.sklearn_model import SklearnModel

class DurationPredictor():
    def __init__(self, model: SklearnModel):
        self.model=model

    def load(self)->None:
        """Delegates loading to the underlying model."""
        self.model.load()

    def predict_one(self,feature:dict[str:any])->float:
        PU=feature["PULocationID"]
        DO=feature["DOLocationID"]
        distance=feature["trip_distance"]

        pu_do=build_feature(PU,DO)

        df={
            "PU_DO":pu_do,
            "trip_distance":distance
        }

        return self.model.predict_one(df)

    def predict_batch(self,features:list[dict[str:any]])->list[float]:
        model_features = []

        for feature in features:
            pu = feature["PULocationID"]
            do = feature["DOLocationID"]
            distance = feature["trip_distance"]

            pu_do = build_feature(pu, do)

            model_features.append(
                {
                    "PU_DO": pu_do,
                    "trip_distance": distance,
                }
            )

        return self.model.predict_batch(model_features)