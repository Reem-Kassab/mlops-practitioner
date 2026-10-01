# MLOps Practitioner — Taxi Duration Prediction API

A production-oriented machine learning service that predicts NYC taxi trip duration from pickup/drop-off locations and trip distance. This project takes a notebook-based scikit-learn model and turns it into a structured Python package, a tested FastAPI service, and a multi-stage non-root Docker image. The service includes configuration management, structured JSON logging with correlation IDs, model serialization in both Pickle and ONNX, prediction-parity testing, latency benchmarking, Docker Compose support, and automated test coverage.

## What This Project Does

The model predicts taxi trip duration in **minutes** using:

* `PULocationID`
* `DOLocationID`
* `trip_distance`

The main engineered categorical feature is:

```text
PU_DO = PULocationID + "_" + DOLocationID
```

The production model is a scikit-learn `LinearRegression` pipeline using:

* `OneHotEncoder(handle_unknown="ignore")` for `PU_DO`
* `trip_distance` passed through as a numerical feature

The service exposes both single and batch prediction endpoints.

---

## Quickstart

You can run the published Docker image without cloning the repository.

### 1. Pull the image

```bash
docker pull reemkassab/prodml-api:0.1.0
```

### 2. Start the API

```bash
docker run --rm -d --name prodml-api -p 8000:8000 reemkassab/prodml-api:0.1.0
```

### 3. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID":138,"DOLocationID":161,"trip_distance":2.5}'
```

Example response:

```json
{
  "prediction": 36.59756851196289,
  "model_version": "v0.1.0",
  "correlation_id": "739e2ca8-147f-462b-8c6e-c7d7bbb4d951",
  "latency_ms": 18.249
}
```

The exact prediction latency and correlation ID will vary between requests and environments.

To stop the container:

```bash
docker stop prodml-api
```

---

## API

The service exposes four endpoints.

| Method | Endpoint         | Description                                                                  |
| ------ | ---------------- | ---------------------------------------------------------------------------- |
| `GET`  | `/health`        | Returns service health and confirms that the model is loaded                 |
| `GET`  | `/metadata`      | Returns model version, training date, features, framework, and artifact hash |
| `POST` | `/predict`       | Returns a prediction for one trip                                            |
| `POST` | `/predict/batch` | Returns predictions for multiple trips                                       |

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

### Health

```bash
curl http://localhost:8000/health
```

Example:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Metadata

```bash
curl http://localhost:8000/metadata
```

### Single Prediction

Request:

```json
{
  "PULocationID": 138,
  "DOLocationID": 161,
  "trip_distance": 2.5
}
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID":138,"DOLocationID":161,"trip_distance":2.5}'
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "trips": [
      {
        "PULocationID": 138,
        "DOLocationID": 161,
        "trip_distance": 2.5
      },
      {
        "PULocationID": 100,
        "DOLocationID": 230,
        "trip_distance": 5.0
      }
    ]
  }'
```

---

## Model Performance

The baseline and production pipeline were evaluated using the same validation setup.

| Metric |          Result |
| ------ | --------------: |
| MAE    |  9.5424 minutes |
| RMSE   | 63.0777 minutes |

The production refactor preserves the model logic and validation performance while moving the implementation from notebook code into a reusable package and service.

---

## Model Serialization

The trained model is stored in two serialization formats:

```text
models/artifacts/taxi_duration.pkl
models/artifacts/taxi_duration.onnx
```

### Pickle

Pickle is retained as the native scikit-learn artifact and reference implementation.

**Important:** Pickle can execute arbitrary code during loading. Only load Pickle artifacts from trusted sources.

### ONNX

ONNX is used by the production inference path through ONNX Runtime.

A parity test runs the same 500 validation rows through both formats and verifies that predictions agree within:

```text
absolute tolerance = 1e-4
```

The parity test passes.

### Serialization Benchmark

Benchmark performed on the same 500 validation rows:

| Format                | Mean latency | P95 latency |
| --------------------- | -----------: | ----------: |
| Pickle / scikit-learn |   20.3285 ms |  51.4133 ms |
| ONNX Runtime          |    6.5792 ms |   8.3110 ms |

In this benchmark, ONNX Runtime had approximately:

* **3.09× lower mean latency**
* **6.19× lower P95 latency**

These numbers are specific to this experiment and environment; they are not treated as universal performance guarantees.

---

## Project Architecture

```text
Client
  |
  v
FastAPI
  |
  +--> /health
  |
  +--> /metadata
  |
  +--> /predict
  |
  +--> /predict/batch
          |
          v
   DurationPredictor
          |
          v
      ONNXModel
          |
          v
    ONNX Runtime
          |
          v
      Prediction
```

The model is loaded once during application startup rather than during every request.

---

## Logging

The service uses structured JSON logging.

Each request can be traced using a correlation ID.

The logging design includes:

* `DEBUG` logs for feature information
* `INFO` logs for successful predictions and latency
* `WARNING` logs for inputs outside the expected training range
* `ERROR` logs for model loading and validation failures
* correlation IDs propagated across request processing

There are no `print()` statements in `src/`.

---

## Configuration

Configuration is stored in:

```text
configs/config.yaml
```

Environment variables can override deployment-specific values such as:

```text
MODEL_VERSION
MODEL_TRAINING_DATE
PRODML_PROJECT_ROOT
```

The project avoids hard-coded machine-specific paths.

---

## Docker

The production container uses a multi-stage Docker build.

The runtime container:

* uses Python 3.11 slim
* runs as the non-root `appuser`
* contains only the runtime application and required artifacts
* exposes port `8000`
* includes a Docker healthcheck
* loads configuration and model artifacts at startup

### Build locally

```bash
docker build -f docker/Dockerfile -t prodml-api:0.1.0 .
```

### Run locally

```bash
docker run --rm -p 8000:8000 prodml-api:0.1.0
```

### Docker Compose

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## Docker Image Comparison

The final multi-stage image was compared against the single-stage implementation.

| Image        | Content size |
| ------------ | -----------: |
| Multi-stage  |       296 MB |
| Single-stage |       305 MB |

The Docker CLI also reported approximately 1.3 GB and 1.34 GB respectively under its disk-usage view. Content size is used as the primary comparison here.

### `.dockerignore` Experiment

The multi-stage image was also built with `.dockerignore` temporarily removed.

| Build                   | Content size |
| ----------------------- | -----------: |
| With `.dockerignore`    |       296 MB |
| Without `.dockerignore` |       296 MB |

No measurable final image-size difference was observed for this project because the Dockerfile copies only the required application, configuration, and model artifacts into the image.

`.dockerignore` is still retained because it prevents unnecessary files from entering the build context and protects the build from accidentally including development files such as virtual environments, caches, tests, raw data, and notebooks.

---

## Testing

The project uses `pytest` with coverage enforcement.

The test suite covers:

* feature engineering
* prediction behaviour
* API endpoints
* serialization parity
* evaluation
* model export
* model adapters
* inference pipeline

Current coverage:

```text
443 statements
114 missed
74.27% coverage
```

The project therefore passes the required **70% coverage gate**.

Run the tests with:

```bash
uv run pytest
```

---

## Development Setup

Create or activate the project environment, then install the package and development dependencies.

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

Run the API locally:

```bash
uv run uvicorn prodml.api.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000/docs
```

---

## Repository Structure

```text
mlops-practitioner/
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
├── models/
│   └── artifacts/
│       ├── taxi_duration.pkl
│       └── taxi_duration.onnx
│
├── notebooks/
│
├── reports/
│   └── module-1.md
│
├── configs/
│   └── config.yaml
│
├── scripts/
│   └── benchmark_serialization.py
│
├── src/
│   └── prodml/
│       ├── api/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── pipelines/
│       └── utils/
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_evaluate.py
│   ├── test_export.py
│   ├── test_features.py
│   ├── test_inference_pipeline.py
│   ├── test_model_adapters.py
│   ├── test_predict.py
│   ├── test_serialization.py
│   └── test_serialization.py
│
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.single-stage
│   └── docker-compose.yml
│
├── .dockerignore
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Module 1 Outcome

This module converts a notebook-based ML experiment into a production-oriented service with:

* a structured, pip-installable Python package
* separated data, feature, model, pipeline, and API layers
* a reusable model interface
* startup model loading
* FastAPI prediction endpoints
* Pydantic validation
* structured JSON logging
* correlation IDs
* Pickle and ONNX model artifacts
* serialization parity testing
* measured inference latency
* pytest coverage above 70%
* multi-stage Docker packaging
* non-root container execution
* Docker Compose deployment

The next module builds on this foundation by adding experiment tracking, model registry, data versioning, continuous training, CI/CD, and infrastructure automation.
