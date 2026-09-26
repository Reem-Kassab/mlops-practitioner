import time
import joblib
import pandas as pd
from taxi_duration.api.main import predictor

pkl_model = joblib.load("models/baseline_linear_pipeline.pkl")
dummy_df_pkl = pd.DataFrame([{"trip_distance": 2.5, "PU_DO": "236_239"}])

start = time.perf_counter()
for _ in range(100):
    pkl_model.predict(dummy_df_pkl)
pkl_time = (time.perf_counter() - start) / 100

predictor.load()
dummy_payload_df = pd.DataFrame([{"PULocationID": 236, "DOLocationID": 239, "trip_distance": 2.5}])

start = time.perf_counter()
for _ in range(100):
    predictor.predict_batch(dummy_payload_df)
onnx_time = (time.perf_counter() - start) / 100

print(f"Pickle Latency: {pkl_time * 1000:.4f} ms")
print(f"ONNX Latency:   {onnx_time * 1000:.4f} ms")
print(f"ONNX is {pkl_time / onnx_time:.2f}x faster!")