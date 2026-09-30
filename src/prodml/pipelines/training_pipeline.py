from prodml.data.ingestion import load_data,split_data
from prodml.data.preprocessing import clean_data
from prodml.data.validation import validate_distance,validate_no_infinite_values,validate_no_missing_values,validate_positive_duration,validate_required_columns
from prodml.features.build_features import build_feature,get_preprocessor
from prodml.models.train import train_model,save_model
from prodml.utils.io import get_raw_data_path,get_model_artifact_path

def run_training():
    #getting the paths
    data_path=get_raw_data_path()
    model_path=get_model_artifact_path()

    #loading the data
    df=load_data(data_path=data_path)
    #check the columns
    validate_required_columns(df=df)
    #clean the data
    df=clean_data(df=df)
    #check the data valuse
    validate_positive_duration(duration=df["duration"])
    validate_no_missing_values(df=df,columns=["PULocationID","DOLocationID"])
    validate_distance(distance=df["trip_distance"])
    #feature engineering
    df["PU_DO"]=build_feature(df["PULocationID"],df["DOLocationID"])
    #feature selection 
    X = df[["PU_DO", "trip_distance"]]
    y = df["duration"]
    #data splitting
    x_train,x_test,y_train,y_test=split_data(X=X,Y=y)
    #building and train the model
    processor=get_preprocessor()
    model=train_model(x_train,y_train,processor)
    #save the model
    save_model(model,model_path)

    return model

if __name__ == "__main__":
    run_training()