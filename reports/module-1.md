# Module 1 — From Notebook to Production-Ready Service

## 1. Overview

This module transforms a notebook-based machine learning experiment into a production-oriented ML service.

The original project was a notebook implementing a NYC TLC green taxi trip-duration prediction model. The notebook was refactored into a pip-installable Python package with separated modules for data ingestion, preprocessing, validation, feature engineering, model training, evaluation, inference, configuration, and API serving.

The final system exposes the model through a FastAPI service, loads the model once at application startup, uses structured JSON logging with correlation IDs, supports both Pickle and ONNX model artifacts, verifies serialization parity, benchmarks inference latency, runs inside a multi-stage non-root Docker container, and is covered by a pytest suite with a coverage gate above 70%.

The target is taxi trip duration in minutes.

---

## 2. Problem Definition

### Target

```text
duration = lpep_dropoff_datetime - lpep_pickup_datetime
```

The resulting target is expressed in minutes.

### Input Features

The production model uses:

* `PULocationID`
* `DOLocationID`
* `trip_distance`

A categorical pickup-to-dropoff feature is engineered as:

```text
PU_DO = PULocationID + "_" + DOLocationID
```

### Model

The production model is a scikit-learn `LinearRegression` pipeline.

Preprocessing:

* `PU_DO` → `OneHotEncoder(handle_unknown="ignore")`
* `trip_distance` → passed through unchanged

The model is wrapped behind a model interface so that the API and inference pipeline are separated from the underlying serialization format.

---

## 3. Baseline vs Production Results

The original notebook implementation and the refactored production implementation use the same model logic and validation setup.

| Metric | Baseline / Notebook |  Production |
| ------ | ------------------: | ----------: |
| MAE    |          9.5424 min |  9.5424 min |
| RMSE   |         63.0777 min | 63.0777 min |

The refactoring therefore preserved the model's validation performance while changing the surrounding engineering architecture.

The main improvement in this module is not a change in model quality. It is the conversion of the notebook into a reproducible, testable, deployable software component.

---

## 4. Project Architecture

The project follows a layered structure:

```text
mlops-practitioner/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
│   └── artifacts/
├── notebooks/
├── reports/
│   └── module-1.md
├── configs/
│   └── config.yaml
├── scripts/
│   └── benchmark_serialization.py
├── src/
│   └── prodml/
│       ├── api/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── pipelines/
│       └── utils/
├── tests/
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.single-stage
│   └── docker-compose.yml
├── .dockerignore
├── pyproject.toml
├── uv.lock
└── README.md
```

The main engineering boundary is the model interface:

```text
API
 ↓
Inference Pipeline
 ↓
DurationPredictor
 ↓
ModelBase
 ↓
ONNXModel / SklearnModel
```

This allows the API and prediction logic to remain independent from the underlying model serialization mechanism.

---

## 5. Configuration

Project configuration is stored in:

```text
configs/config.yaml
```

The configuration layer provides paths and training settings without hardcoding machine-specific absolute paths.

The application also supports environment-specific values such as:

```text
PRODML_PROJECT_ROOT
MODEL_VERSION
MODEL_TRAINING_DATE
```

Container execution uses:

```text
PRODML_PROJECT_ROOT=/app
```

This keeps local and container paths separate while allowing the same codebase to run in both environments.

---

## 6. Data Pipeline

The production data flow is separated into explicit stages:

```text
Raw Parquet
    ↓
Data Ingestion
    ↓
Required-column Validation
    ↓
Data Cleaning
    ↓
Value Validation
    ↓
Feature Engineering
    ↓
Train / Test Split
    ↓
Model Training
    ↓
Evaluation
    ↓
Model Serialization
```

Validation includes checks for:

* required columns
* missing values
* infinite values
* invalid duration values
* invalid distance values

This prevents invalid training data from silently reaching the model.

---

## 7. Feature Engineering

The main derived feature is:

```text
PU_DO = PULocationID + "_" + DOLocationID
```

For example:

```text
PULocationID = 138
DOLocationID = 161

PU_DO = "138_161"
```

The feature transformation is shared by training and inference so that the same representation is used in both paths.

The preprocessing transformer uses:

```text
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)
```

This allows the service to handle pickup/dropoff combinations that were not present during training without raising an encoding error.

---

## 8. Model Abstraction

The project introduces a model interface with the following responsibilities:

```text
load()
predict_one()
predict_batch()
```

The interface is implemented by:

```text
SklearnModel
ONNXModel
```

`DurationPredictor` uses composition rather than inheriting from a specific model implementation.

This means the prediction layer does not need to know whether the model is backed by Pickle/scikit-learn or ONNX Runtime.

The design creates a stable seam between the API and model implementation and makes it possible to change the runtime without rewriting the API layer.

---

## 9. Training Pipeline

The training pipeline performs:

1. Data loading
2. Required-column validation
3. Cleaning
4. Value validation
5. Feature engineering
6. Feature/target selection
7. Train/test splitting
8. Model training
9. Evaluation
10. Pickle serialization
11. ONNX export
12. Serialization validation sample generation

A fixed validation sample of 500 rows is saved for serialization parity testing.

---

## 10. Model Artifacts

The project produces two model artifacts:

```text
models/artifacts/taxi_duration.pkl
models/artifacts/taxi_duration.onnx
```

### Pickle

Pickle represents the native scikit-learn model artifact.

It is retained as the reference implementation for comparison and validation.

### ONNX

ONNX provides a portable model representation that can be executed using ONNX Runtime.

The production inference pipeline uses the ONNX artifact.

### Security consideration

Pickle must only be loaded from a trusted source because loading an arbitrary Pickle file can execute Python code.

For that reason, the Pickle artifact is treated as a trusted internal artifact rather than an arbitrary external model format.

---

## 11. Serialization Parity

The project uses a fixed 500-row validation sample to compare predictions from the Pickle and ONNX versions.

The parity condition is:

```text
np.allclose(pred_pickle, pred_onnx, atol=1e-4)
```

### Result

```text
Validation rows: 500
Tolerance: atol=1e-4
Result: PASS
```

This confirms that converting the model to ONNX did not materially change its predictions on the validation sample.

---

## 12. Serialization Benchmark

Both serialization formats were benchmarked using the same 500 validation rows.

| Format                | Mean latency | P95 latency |
| --------------------- | -----------: | ----------: |
| Pickle / scikit-learn |   20.3285 ms |  51.4133 ms |
| ONNX Runtime          |    6.5792 ms |   8.3110 ms |

Observed differences in this benchmark:

```text
Mean latency speedup:
20.3285 / 6.5792 ≈ 3.09x

P95 latency reduction:
51.4133 / 8.3110 ≈ 6.19x
```

These measurements describe this particular local benchmark and environment. They are not treated as universal performance guarantees.

Based on the measured result, ONNX Runtime is used for the production inference path while Pickle remains useful as the reference artifact.

---

## 13. Serialization Format Comparison

| Format   | Human-readable | Cross-language   | Schema-enforced | Safe to load from untrusted source |
| -------- | -------------- | ---------------- | --------------- | ---------------------------------- |
| JSON     | Yes            | Yes              | No              | Yes                                |
| Protobuf | No             | Yes              | Yes             | Yes                                |
| Pickle   | No             | Primarily Python | No              | No                                 |
| ONNX     | No             | Yes              | Model-defined   | Yes*                               |

`*` ONNX is not equivalent to executing arbitrary Python Pickle code, but production systems should still treat model artifacts as controlled inputs.

---

## 14. Structured Logging

The project implements structured JSON logging.

Each request receives a correlation ID that is propagated through the request lifecycle.

Logging levels are used according to event severity:

| Level   | Example                                       |
| ------- | --------------------------------------------- |
| DEBUG   | Feature vector information                    |
| INFO    | Prediction served and latency                 |
| WARNING | Input outside the expected training range     |
| ERROR   | Model loading failure or validation rejection |

The service uses `contextvars` for correlation-ID propagation.

Startup logs may use:

```text
correlation_id = "-"
```

while request-related logs carry the request correlation ID.

No `print()` statements remain in `src/`.

---

## 15. FastAPI Service

The FastAPI application exposes four endpoints:

| Method | Endpoint         | Purpose                           |
| ------ | ---------------- | --------------------------------- |
| GET    | `/health`        | Confirms that the model is loaded |
| GET    | `/metadata`      | Returns model metadata            |
| POST   | `/predict`       | Single prediction                 |
| POST   | `/predict/batch` | Batch prediction                  |

### `/health`

The endpoint checks model availability rather than only checking whether the process is alive.

Example:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `/metadata`

Returns information such as:

* model version
* training date
* feature names
* framework
* artifact SHA-256 hash

### `/predict`

Accepts:

```json
{
  "PULocationID": 138,
  "DOLocationID": 161,
  "trip_distance": 2.5
}
```

An observed prediction from the production container was approximately:

```text
36.5976 minutes
```

### `/predict/batch`

Accepts multiple trips and returns a prediction for each input.

---

## 16. Startup Model Loading

The model is loaded once when the FastAPI application starts.

The inference object is stored at application scope and reused across requests.

The model is therefore not instantiated inside the request handler.

This avoids repeated model loading and keeps request processing focused on validation, feature preparation, inference, and response construction.

---

## 17. API Validation

The API uses Pydantic schemas to validate requests.

Invalid inputs are rejected before inference.

Examples of invalid values include:

```text
trip_distance <= 0
trip_distance outside the configured maximum
missing required input fields
invalid field types
```

Validation errors return a structured HTTP 422 response instead of leaking internal stack traces.

Unexpected internal errors are handled by the application's exception handling layer.

---

## 18. Inference Pipeline

The production inference path is:

```text
HTTP Request
    ↓
Pydantic Validation
    ↓
FastAPI Route
    ↓
DurationPredictor
    ↓
Feature Construction
    ↓
ONNXModel
    ↓
ONNX Runtime
    ↓
Prediction
    ↓
Structured Response + Correlation ID
```

ONNX Runtime's `InferenceSession` is initialized once and reused for subsequent requests.

---

## 19. Testing

The project uses `pytest` with an enforced coverage threshold.

Tests cover:

* feature engineering
* prediction behaviour
* API endpoints
* model adapters
* evaluation
* model export
* inference pipeline
* serialization parity

The current coverage result is:

```text
Total statements: 443
Missed statements: 114
Coverage: 74.27%
Required: 70%
Result: PASS
```

The project therefore exceeds the required coverage threshold.

The test suite is run with:

```bash
uv run pytest
```

---

## 20. Dockerization

The production image is built using a multi-stage Dockerfile.

### Builder stage

The builder installs the application and its dependencies.

### Runtime stage

The runtime image contains:

* Python runtime
* installed application dependencies
* model artifacts
* configuration files

The container runs as:

```text
appuser
```

rather than as root.

The API listens on:

```text
0.0.0.0:8000
```

A Docker healthcheck targets:

```text
/health
```

---

## 21. Docker Image Size

Two Dockerfile implementations were compared.

| Image        | Content size |
| ------------ | -----------: |
| Multi-stage  |       296 MB |
| Single-stage |       305 MB |

The multi-stage image therefore produced a smaller final content size in this project.

The Docker CLI displayed larger disk-usage values because its disk-usage accounting includes Docker's layer/storage representation.

The primary comparison used above is the image content size.

---

## 22. `.dockerignore` Experiment

The project was built once with `.dockerignore` enabled and once with it temporarily removed.

| Build                   | Content size |
| ----------------------- | -----------: |
| With `.dockerignore`    |       296 MB |
| Without `.dockerignore` |       296 MB |

### Result

No measurable final image-size difference was observed.

The reason is that the Dockerfile explicitly copies only the files required by the image:

```text
src/
models/
configs/
```

Therefore, the extra files excluded by `.dockerignore` do not become part of the final runtime image.

`.dockerignore` is still retained because it keeps the build context clean and prevents unnecessary development files such as:

```text
.venv/
.git/
notebooks/
data/
tests/
__pycache__/
cache files
```

from being sent to the Docker builder.

---

## 23. Docker Compose

The project includes:

```text
docker/docker-compose.yml
```

The Compose configuration provides:

* API service
* port mapping
* environment variables
* model artifact volume
* restart policy
* healthcheck

The service was started successfully with:

```bash
docker compose -f docker/docker-compose.yml up
```

The following behaviours were verified:

* `/health` returned HTTP 200
* `/predict` returned a prediction
* `/predict/batch` returned batch predictions
* the container ran as `appuser`
* Docker healthcheck passed

---

## 24. Deployment Validation

The production Docker image was successfully built and validated locally.

A production container returned:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

A real prediction request also completed successfully.

The container therefore demonstrated the complete path:

```text
Docker container
    ↓
FastAPI startup
    ↓
Model loading
    ↓
Health check
    ↓
Prediction request
    ↓
Model inference
    ↓
JSON response
```

---

## 25. Maturity Self-Assessment

### Current maturity

**Module 1 production-serving stage — a repeatable, containerized ML service with automated testing and structured operational boundaries.**

The repository has moved beyond a notebook-only workflow into a packaged and deployable service with explicit data, feature, model, inference, API, logging, testing, serialization, and containerization boundaries.

### What is missing for the next level

Experiment tracking, model registry, data versioning, continuous training, CI/CD automation, and infrastructure automation are not yet implemented. Module 2 will add these capabilities to make the ML lifecycle more reproducible, auditable, and automated.

---

## 26. Definition of Done

### Engineering

* [x] Python package with `src/` layout
* [x] Clear separation between data, features, training, inference, and API
* [x] Model interface with concrete adapters
* [x] Configuration separated from application code
* [x] No hardcoded machine-specific paths
* [x] Type hints and structured module boundaries

### API

* [x] `/health`
* [x] `/metadata`
* [x] `/predict`
* [x] `/predict/batch`
* [x] Pydantic request/response validation
* [x] Model loaded at startup
* [x] Error handling implemented

### Logging

* [x] Structured JSON logs
* [x] Correlation IDs
* [x] Appropriate log levels
* [x] Zero `print()` statements in `src/`

### Serialization

* [x] Pickle artifact
* [x] ONNX artifact
* [x] 500-row parity test
* [x] `atol=1e-4`
* [x] Benchmark for mean latency
* [x] Benchmark for P95 latency
* [x] JSON / Protobuf / Pickle / ONNX comparison
* [x] Pickle security warning documented

### Testing

* [x] Pytest suite
* [x] Fixtures
* [x] API tests
* [x] Serialization tests
* [x] Model adapter tests
* [x] Export tests
* [x] Evaluation tests
* [x] Inference pipeline tests
* [x] Coverage gate ≥ 70%
* [x] Current coverage: 74.27%

### Docker

* [x] Multi-stage Dockerfile
* [x] Single-stage comparison
* [x] `.dockerignore`
* [x] `.dockerignore` size experiment
* [x] Docker Compose
* [x] Non-root runtime user
* [x] Container healthcheck
* [x] Local production container validated
* [ ] Public Docker Hub image successfully pushed and pull-tested

### Documentation and release

* [x] Three-command README
* [x] Real prediction example
* [x] Repository tree documented
* [x] Module report completed
* [ ] Pull request opened and reviewed
* [ ] Pull request merged into `main`
* [ ] `v0.1.0` release tag created

---

## 27. Known Limitations and Next Steps

The service is intentionally limited to the scope of Module 1.

The current system does not yet include:

* experiment tracking
* model registry
* data versioning
* automated retraining
* continuous integration
* continuous deployment
* infrastructure as code
* production monitoring
* drift detection
* load testing
* advanced serving runtimes

These are intentionally deferred to later modules.

Module 2 will build the automation and reproducibility layer around the production foundation created here.

---

## 28. Final Module 1 Summary

Module 1 converted a notebook-based taxi-duration model into a production-oriented ML service.

The final system includes:

```text
Notebook
   ↓
Python Package
   ↓
Training Pipeline
   ↓
Model Artifacts
   ├── Pickle
   └── ONNX
   ↓
Parity + Benchmark
   ↓
Inference Pipeline
   ↓
FastAPI
   ├── /health
   ├── /metadata
   ├── /predict
   └── /predict/batch
   ↓
Structured JSON Logging
   ↓
Pytest Coverage Gate
   ↓
Multi-stage Non-root Docker
   ↓
Docker Compose
```

The baseline prediction quality was preserved while the engineering system around the model was significantly strengthened.

The next stage is Module 2: experiment tracking, model registry, data versioning, continuous training, CI/CD, and infrastructure automation.
