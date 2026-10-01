import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error

def evaluate_model(model:Pipeline,x_test:pd.DataFrame,y_test:pd.Series)->tuple[float,float]:
    """calculating the MAE and the RMSE"""
    predictions= model.predict(x_test)
    mae=mean_absolute_error(y_test,predictions)
    rmse=np.sqrt(mean_squared_error(y_test,predictions))

    return mae,rmse

