import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(
    os.getenv(
        "PRODML_PROJECT_ROOT",
        Path.cwd(),
    )
)
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"


class ProjectConfig(BaseModel):
    name: str
    version: str


class DataConfig(BaseModel):
    raw_path: str
    processed_path: str
    validation_sample_name: str


class ModelConfig(BaseModel):
    artifact_dir: str
    artifact_name: str
    onnx_artifact_name: str


class FeaturesConfig(BaseModel):
    categorical: list[str]
    numerical: list[str]


class TrainingConfig(BaseModel):
    test_size: float
    random_state: int

class ApiConfig(BaseModel):
    host: str
    port: int


class Settings(BaseSettings):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    features: FeaturesConfig
    training: TrainingConfig
    api: ApiConfig

    model_config = SettingsConfigDict(
        env_prefix="PRODML_",
        env_nested_delimiter="__",
        extra="ignore",
    )



def load_settings() -> Settings:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        yaml_config: dict[str, Any] = yaml.safe_load(file) or {}

    return Settings(**yaml_config)


settings = load_settings()