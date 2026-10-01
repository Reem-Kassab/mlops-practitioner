import math

from prodml.models.predictor import DurationPredictor


def test_prediction_returns_float(predictor: DurationPredictor,sample_features: dict):
    prediction = predictor.predict_one(sample_features)

    assert isinstance(prediction, float)


def test_prediction_is_sane(predictor: DurationPredictor,sample_features: dict,):
    prediction = predictor.predict_one(sample_features)

    assert math.isfinite(prediction)
    assert 0 < prediction < 100


def test_prediction_is_deterministic(predictor: DurationPredictor,sample_features: dict,):
    prediction_1 = predictor.predict_one(sample_features)

    prediction_2 = predictor.predict_one(sample_features)

    assert prediction_1 == prediction_2


def test_batch_prediction(predictor: DurationPredictor,):
    features = [
        {
            "PULocationID": 138,
            "DOLocationID": 161,
            "trip_distance": 2.5,
        },
        {
            "PULocationID": 100,
            "DOLocationID": 186,
            "trip_distance": 4.1,
        },
    ]

    predictions = predictor.predict_batch(features)

    assert isinstance(predictions, list)
    assert len(predictions) == 2

    assert all(isinstance(prediction, float)
        for prediction in predictions)