from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

import prodml.api.main as api_main
from prodml.api.main import app
from prodml.features.build_features import get_preprocessor
from prodml.models.base import ModelBase
from prodml.models.predictor import DurationPredictor
from prodml.models.train import train_model


class InMemoryModel(ModelBase):
    """
    Small ModelBase implementation used only by unit tests.

    It wraps a fitted sklearn pipeline without loading an
    artifact from disk.
    """

    def __init__(self, model: Pipeline):
        self.model = model
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def predict_one(self, features: dict) -> float:
        if not self.loaded:
            raise RuntimeError("Model is not loaded.")

        features_df = pd.DataFrame([features])
        prediction = self.model.predict(features_df)

        return float(prediction[0])

    def predict_batch(
        self,
        features: list[dict],
    ) -> list[float]:
        if not self.loaded:
            raise RuntimeError("Model is not loaded.")

        features_df = pd.DataFrame(features)
        predictions = self.model.predict(features_df)

        return predictions.tolist()


class FakePredictor:
    """
    Fake predictor used by API tests.

    The API should be tested without depending on a real
    training run or production artifact.
    """

    def __init__(self):
        self.model = SimpleNamespace()

    def predict_one(self, features: dict) -> float:
        return 12.5

    def predict_batch(
        self,
        features: list[dict],
    ) -> list[float]:
        return [
            12.5 + index
            for index in range(len(features))
        ]


@pytest.fixture
def sample_features() -> dict:
    """Representative request for a single prediction."""

    return {
        "PULocationID": 138,
        "DOLocationID": 161,
        "trip_distance": 2.5,
    }


@pytest.fixture(scope="session")
def trained_model() -> Pipeline:
    """
    Small deterministic model used across the test session.

    This avoids retraining for every test.
    """

    X_train = pd.DataFrame(
        {
            "PU_DO": [
                "1_2",
                "1_3",
                "2_3",
                "2_4",
                "3_4",
                "3_5",
            ],
            "trip_distance": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
        }
    )

    y_train = pd.Series(
        [
            10.0,
            12.0,
            16.0,
            20.0,
            24.0,
            28.0,
        ]
    )

    preprocessor = get_preprocessor()

    return train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
    )


@pytest.fixture
def predictor(trained_model: Pipeline) -> DurationPredictor:
    """Application-level predictor backed by the test model."""

    model = InMemoryModel(trained_model)

    predictor = DurationPredictor(model)
    predictor.load()

    return predictor


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """
    FastAPI TestClient.

    The real create_predictor() is replaced so API tests
    do not depend on loading a production artifact.
    """

    fake_predictor = FakePredictor()

    monkeypatch.setattr(
        api_main,
        "create_predictor",
        lambda: fake_predictor,
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client