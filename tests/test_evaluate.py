import numpy as np
import pandas as pd

from prodml.models.evaluate import evaluate_model


def test_evaluate_model_returns_metrics(
    trained_model,
):
    x_test = pd.DataFrame(
        {
            "PU_DO": [
                "1_2",
                "2_3",
                "3_4",
            ],
            "trip_distance": [
                1.5,
                2.5,
                4.0,
            ],
        }
    )

    y_test = pd.Series(
        [
            11.0,
            15.0,
            22.0,
        ]
    )

    mae, rmse = evaluate_model(
        trained_model,
        x_test,
        y_test,
    )

    assert isinstance(mae, float)
    assert isinstance(rmse, float)

    assert np.isfinite(mae)
    assert np.isfinite(rmse)

    assert mae >= 0
    assert rmse >= 0

    # RMSE should never be smaller than MAE
    assert rmse >= mae