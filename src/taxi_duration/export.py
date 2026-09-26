import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from taxi_duration.config import settings
import logging
logger = logging.getLogger(__name__)


def export_to_onnx(pkl_model_path: str, onnx_model_path: str):
    """
    Converts the scikit-learn pipeline to ONNX format.
    """

    pipeline = joblib.load(pkl_model_path)
    
    #identify the data into onxx
    initial_types = [
        ('PU_DO', StringTensorType([None, 1])),
        ('trip_distance', FloatTensorType([None, 1]))
    ]
    
    onnx_model = convert_sklearn(
        pipeline, 
        initial_types=initial_types,
        target_opset=12 
    )
    
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    logger.info(f" ONNX model saved to: {onnx_model_path}")


def test_parity(pkl_model_path: str, onnx_model_path: str):
    """
    Ensures both models produce the exact same output.
    """
    pipeline = joblib.load(pkl_model_path)
    sess = rt.InferenceSession(onnx_model_path)
    
    sample_df = pd.DataFrame({
        "PU_DO": ["10_50", "100_200"],
        "trip_distance": [3.5, 10.2]
    })
    sklearn_preds = pipeline.predict(sample_df)

    onnx_inputs = {
        'PU_DO': sample_df['PU_DO'].to_numpy().reshape(-1, 1).astype(str),
        'trip_distance': sample_df['trip_distance'].to_numpy().reshape(-1, 1).astype(np.float32)
    }
    onnx_preds = sess.run(None, onnx_inputs)[0].flatten()
    
    logger.info("\n--- Parity Test ---")
    logger.info(f"Sklearn Predictions: {sklearn_preds}")
    logger.info(f"ONNX Predictions:    {onnx_preds}")
    np.testing.assert_allclose(sklearn_preds, onnx_preds, rtol=1e-4, atol=1e-4)
    logger.info("Parity Test Passed! Both models produce the exact same predictions.")


if __name__ == "__main__":
    ONNX_PATH = settings.model_save_path.replace(".pkl", ".onnx")
    
    logger.info(" Starting ONNX Export...")
    export_to_onnx(settings.model_save_path, ONNX_PATH)
    
    logger.info("\n Running Parity Test...")
    test_parity(settings.model_save_path, ONNX_PATH)