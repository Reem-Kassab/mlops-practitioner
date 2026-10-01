import time
from pathlib import Path

import numpy as np
import pandas as pd

from prodml.models.onnx_model import ONNXModel
from prodml.models.sklearn_model import SklearnModel
from prodml.utils.io import (
    get_model_artifact_path,
    get_onnx_artifact_path,
    get_validation_sample_path,
)


WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


def benchmark_model(model, features: list[dict]) -> tuple[float, float]:
    """Benchmark batch prediction latency in milliseconds."""

    for _ in range(WARMUP_RUNS):
        model.predict_batch(features)

    latencies = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        model.predict_batch(features)

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    return mean_latency, p95_latency


def main() -> None:
    validation_path = get_validation_sample_path()

    validation_data = pd.read_parquet(
        validation_path
    )

    features = validation_data.to_dict(
        orient="records"
    )

    sklearn_model = SklearnModel(
        get_model_artifact_path()
    )
    sklearn_model.load()

    onnx_model = ONNXModel(
        get_onnx_artifact_path()
    )
    onnx_model.load()

    sklearn_mean, sklearn_p95 = benchmark_model(
        sklearn_model,
        features,
    )

    onnx_mean, onnx_p95 = benchmark_model(
        onnx_model,
        features,
    )

    print("\nSerialization Benchmark")
    print("-" * 50)
    print(f"Rows per prediction batch: {len(features)}")
    print()
    print("Format       Mean Latency (ms)    P95 Latency (ms)")
    print("-" * 50)
    print(
        f"Pickle       {sklearn_mean:>15.4f}"
        f"    {sklearn_p95:>15.4f}"
    )
    print(
        f"ONNX         {onnx_mean:>15.4f}"
        f"    {onnx_p95:>15.4f}"
    )


if __name__ == "__main__":
    main()