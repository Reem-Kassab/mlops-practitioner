from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):

    data_path: str = Field(
        default=str(PROJECT_ROOT / "data/green_tripdata_2026-01.parquet"), 
        env="TAXI_DATA_PATH"
    )
    model_save_path: str = Field(
        default=str(PROJECT_ROOT / "models/baseline_linear_pipeline.pkl"), 
        env="TAXI_MODEL_PATH"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()