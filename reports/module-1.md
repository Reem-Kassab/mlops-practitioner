# Module 1 Report: Taxi Trip Duration Predictor

## 1. Model Performance
- **Baseline MAE:** 9.5424 minutes
- **RMSE:** 63.0777 minutes

## 2. Latency Comparison
*Note: Inference tests were run locally on single-row data. While ONNX shows a higher latency here due to Python session overhead for single rows, it scales significantly better in production batch processing.*
- **Pickle Latency:** 10.9965 ms per prediction
- **ONNX Latency:** 24.5053 ms per prediction

## 3. Docker Image Optimization
By implementing a multi-stage Docker build, we successfully stripped out build tools and dependency caches (`uv` artifacts) from the final production image.
- **Single-stage image size:** 1.25 GB
- **Multi-stage image size:** 1.13 GB
- **Size Reduction:** ~9.6% reduction in total footprint.

## 4. Serialization Format Comparison

| Format | Human-Readable | Cross-Language | Schema-Enforced | Safe from Untrusted Sources |
|---|---|---|---|---|
| **JSON** | Yes | Yes | No | Yes |
| **Protobuf** | No | Yes | Yes | Yes |
| **Pickle** | No | No (Python only) | No | **No (Executes arbitrary code)** |
| **ONNX** | No | Yes | Yes | Yes |

**Chosen Format for this Service: ONNX**
We transitioned from Pickle to ONNX primarily for security and interoperability. Pickle is inherently insecure—loading a compromised `.pkl` file can execute arbitrary malicious code. ONNX eliminates this fatal security flaw, enforces a strict schema, and allows our model to be served in cross-language environments (e.g., C++ or Rust backends) in the future.
