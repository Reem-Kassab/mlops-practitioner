import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_feature(pu:pd.Series|int,do:pd.Series|int)->pd.Series:
    """This function ctreate the main feature"""
    PU_DO = pu.astype(str) + "_" + do.astype(str)
    return PU_DO


def get_preprocessor()->ColumnTransformer:
    """creating the processor transformar for the model input"""
    preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ["PU_DO"]),
        ("numerical", "passthrough",["trip_distance"]),
    ]
    )
    return preprocessor