from prodml.utils.config import PROJECT_ROOT, settings

def get_raw_data_path() -> str:
    """getting the path of the data"""
    data_path = (
        PROJECT_ROOT
        / settings.data.raw_path
        / "green_tripdata_2026-01.parquet"
     )
    return data_path

def get_model_artifact_path()->str:
    """gitting the path of the model"""
    model_path = (
        PROJECT_ROOT
        / settings.model.artifact_dir
        / settings.model.artifact_name
    )
    return model_path