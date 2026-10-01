import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train:pd.DataFrame,y_train:pd.Series,preprocessor:ColumnTransformer)->Pipeline:
    """Build the full pipeline and train the model"""
    model=Pipeline([
        ("preprocessor",preprocessor),
        ("regressor",LinearRegression()),
    ])

    model.fit(X_train,y_train)

    return model

def save_model(model:Pipeline,model_path:str)->None:
    """Save the trained model artifact to disk."""
    joblib.dump(model,model_path)