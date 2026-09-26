import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies deterministic feature engineering 
    and selects only the required columns for the baseline model.
    """
    # compining location
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    # data selection
    selected_columns = ["PU_DO", "trip_distance", "duration"] 
    return df[selected_columns].copy()


def get_preprocessor() -> ColumnTransformer:
    """
    Creates and returns the scikit-learn preprocessor pipeline.
    """
    categorical_features = ["PU_DO"]
    numerical_features = ["trip_distance"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )
    
    return preprocessor