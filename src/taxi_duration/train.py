import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from taxi_duration.config import settings
from taxi_duration.data import load_and_clean_data
from taxi_duration.features import build_features, get_preprocessor
from taxi_duration.logging_conf import setup_logging

logger = setup_logging("taxi_duration_train")

def run_training(data_path: str, model_save_path: str):
    logger.info("Loading and cleaning data...")
    df = load_and_clean_data(data_path)
    
    logger.info("Building features (PU_DO)...")
    df = build_features(df)
    
    logger.info("Splitting data...")
    X = df[['trip_distance', 'PU_DO']]
    y = df['duration'] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info("Building preprocessing pipeline...")
    preprocessor = get_preprocessor()
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    
    logger.info("Training the model...")
    pipeline.fit(X_train, y_train)
    
    score = pipeline.score(X_test, y_test)
    logger.info(f"Model R2 Score on test set: {score:.4f}")
    
    logger.info(f"Saving model to {model_save_path}...")
    joblib.dump(pipeline, model_save_path)

def main():
    logger.info("Starting the MLOps Training Pipeline...")
    run_training(
        data_path=settings.data_path,
        model_save_path=settings.model_save_path
    )
    logger.info("Pipeline finished successfully!")

if __name__ == "__main__":
    main()