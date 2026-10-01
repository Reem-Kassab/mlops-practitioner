import pandas as pd
import pytest
from pydantic import ValidationError

from prodml.api.schemas.prediction import PredictionRequest
from prodml.features.build_features import build_feature,get_preprocessor


def test_build_feature():
    pu = pd.Series([138, 100])
    do = pd.Series([161, 186])

    result = build_feature(pu, do)

    expected = pd.Series(
        [
            "138_161",
            "100_186",
        ]
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


@pytest.mark.parametrize(
    ("payload", "error_field"),
    [
        (
            {
                "PULocationID": 138,
                "DOLocationID": 161,
                "trip_distance": 0,
            },
            "trip_distance",
        ),
        (
            {
                "PULocationID": 138,
                "DOLocationID": 161,
                "trip_distance": -1,
            },
            "trip_distance",
        ),
        (
            {
                "PULocationID": 138,
                "DOLocationID": 161,
                "trip_distance": 200,
            },
            "trip_distance",
        ),
    ],
)
def test_invalid_trip_distance_is_rejected(
    payload: dict,
    error_field: str,
):
    with pytest.raises(ValidationError) as exc_info:
        PredictionRequest.model_validate(payload)

    error_text = str(exc_info.value)

    assert error_field in error_text


@pytest.mark.parametrize(
    "missing_field",
    [
        "PULocationID",
        "DOLocationID",
    ],
)
def test_missing_location_is_rejected(
    missing_field: str,
):
    payload = {
        "PULocationID": 138,
        "DOLocationID": 161,
        "trip_distance": 2.5,
    }

    payload.pop(missing_field)

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


def test_unseen_pu_do_pair_is_supported():
    train_features = pd.DataFrame(
        {
            "PU_DO": [
                "138_161",
                "100_186",
            ],
            "trip_distance": [
                2.5,
                4.1,
            ],
        }
    )

    unseen_features = pd.DataFrame(
        {
            "PU_DO": [
                "999_999",
            ],
            "trip_distance": [
                3.2,
            ],
        }
    )

    preprocessor = get_preprocessor()

    preprocessor.fit(train_features)

    transformed = preprocessor.transform(
        unseen_features
    )

    assert transformed.shape[0] == 1