import pandas as pd
from prodml.features.build_features import build_feature,get_preprocessor

def test_build_feature():
    result = build_feature(1, 23)
    assert result == "1_23"