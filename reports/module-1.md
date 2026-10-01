# Module 1 — From Notebook to Production-Ready Service

## 1. Project Overview

This module transforms a notebook-based taxi trip-duration regression model into a production-oriented Python package and inference service.

The model predicts taxi trip duration in minutes using:

* `PU_DO`: pickup-location / dropoff-location pair
* `trip_distance`: trip distance in miles

The project uses the NYC TLC trip-duration regression problem as the reference workload.

The goal of this module is not to optimize the predictive model itself, but to establish the engineering infrastructure around it: packaging, configuration, validation, structured logging, serialization, inference, API serving, testing, and containerization.

The handbook explicitly frames the five mini-projects as layers of one system, with Module 1 turning the notebook into a production-ready service.

---

## 2. Baseline Model

### Dataset

One month of NYC TLC trip data was used for the baseline experiment.

### Target

Trip duration in minutes:

`duration = dropoff_timestamp - pickup_timestamp`

### Features

The baseline model uses:

* `PU_DO`
* `trip_distance`

This follows the handbook's reference baseline definition.

### Data Preparation

The exploratory notebook included:

* column inspection and understanding
* removal of irrelevant/leakage-prone columns
* conversion of pickup/dropoff timestamps
* calculation of trip duration
* removal of physically implausible records
* feature construction for `PU_DO`

The final production package separates these responsibilities into data ingestion, preprocessing, validation, feature engineering, model training, and prediction modules.

---

## 3. Baseline Performance

| Metric | Notebook Baseline | Production Pipeline |
| ------ | ----------------: | ------------------: |
| MAE    |              TODO |                TODO |
| RMSE   |              TODO |                TODO |

The production training pipeline must reproduce the notebook baseline within the tolerance specified by the handbook.

### Interpretation

The baseline MAE was approximately 9.54 minutes, meaning the average absolute prediction error was about 9.5 minutes.

The RMSE was substantially larger than the MAE, indicating that a smaller number of large prediction errors contribute strongly to the squared-error metric.

---

## 4. Production Package Architecture

The notebook was decomposed into a modular Python package under `src/prodml`.

### Data Layer

Responsible for:

* loading Parquet data
* splitting training and validation data
* cleaning data
* validating data quality

### Feature Layer

Responsible for:

* constructing `PU_DO`
* defining the scikit-learn preprocessing transformer

### Model Layer

Responsible for:

* defining the model interface
* training and persisting the sklearn model
* evaluating predictions
* loading the serialized model
* serving single and batch predictions
* supporting the ONNX implementation

### Pipeline Layer

Responsible for orchestration:

* `training_pipeline.py` connects the complete training workflow
* `inference_pipeline.py` connects model loading and inference

### API Layer

Responsible only for HTTP concerns:

* request validation
* response schemas
* routes
* application lifecycle
* dependency injection

This follows the handbook's requirement for clean boundaries between data, features, training, and prediction.

---

## 5. Configuration

Configuration is centralized and validated using Pydantic.

Configuration includes:

* project metadata
* data paths
* model artifact paths
* feature definitions
* training parameters
* API settings

The application does not depend on hardcoded absolute paths.

Environment variables can override configuration values at runtime, following the handbook's `pydantic-settings` requirement.

---

## 6. Structured Logging

The project uses structured JSON logging.

Each log record contains:

* timestamp
* level
* logger
* message
* correlation ID

Logging levels are used according to event importance:

* `DEBUG`: prepared feature vector
* `INFO`: prediction served and latency
* `WARNING`: unusual inputs such as trip distance outside the expected training range
* `ERROR`: model loading failures and validation rejection

The correlation ID will be generated in FastAPI middleware and propagated through the request context.

---

## 7. Serialization

The fitted scikit-learn pipeline is stored in two formats:

```text
models/artifacts/
├── taxi_duration.pkl
└── taxi_duration.onnx
```

The Pickle artifact is the Python/scikit-learn baseline.

The ONNX artifact is executed using ONNX Runtime.

The ONNX model uses a dynamic batch dimension so that inference can operate on different batch sizes.

---

## 8. Pickle vs ONNX Parity

The same 500 validation rows were evaluated with both serialized model formats.

Parity result:

`PASS`

Tolerance:

`atol = 1e-4`

This confirms that the ONNX conversion preserved the model's predictions within the required tolerance.

The handbook explicitly requires prediction parity on 500 validation rows using `np.allclose(..., atol=1e-4)`.

---

## 9. Serialization Latency Benchmark

The same 500-row validation batch was used for the latency benchmark.

| Format                | Mean Latency | P95 Latency |
| --------------------- | -----------: | ----------: |
| Pickle / scikit-learn |   20.3285 ms |  51.4133 ms |
| ONNX / ONNX Runtime   |    6.5792 ms |   8.3110 ms |

The benchmark measured inference execution time and excluded model loading.

### Observations

For this benchmark environment:

* ONNX mean latency was approximately 3.1× lower than the Pickle baseline.
* ONNX p95 latency was approximately 6.2× lower than the Pickle baseline.
* ONNX also showed a smaller gap between mean and p95 latency in this benchmark.

These measurements describe this experiment and environment; they are not a universal performance guarantee.

---

## 10. Serialization Format Comparison

| Format   | Human-readable | Cross-language | Schema-enforced    | Safe to load from an untrusted source         |
| -------- | -------------- | -------------- | ------------------ | --------------------------------------------- |
| JSON     | Yes            | Yes            | No                 | Generally safe for data parsing               |
| Protobuf | No             | Yes            | Yes                | Generally safe for data parsing               |
| Pickle   | No             | No             | No                 | No                                            |
| ONNX     | No             | Yes            | Model graph/schema | Do not treat untrusted model files as trusted |

Pickle is specifically unsafe to load from untrusted sources because loading a pickle can execute arbitrary code. The handbook explicitly requires this warning in the report.

### Serving Decision

The service will use ONNX Runtime for inference because the ONNX artifact passed the required prediction-parity test and showed lower measured mean and p95 batch latency in this benchmark.

---

## 11. API Design

The production service will expose:

| Endpoint         | Method | Purpose                             |
| ---------------- | ------ | ----------------------------------- |
| `/health`        | GET    | Verify the model is actually loaded |
| `/metadata`      | GET    | Return model metadata               |
| `/predict`       | POST   | Predict one trip duration           |
| `/predict/batch` | POST   | Predict multiple trip durations     |

The handbook requires these four endpoints and requires the model to be loaded once during application startup, rather than inside request handlers.

---

## 12. Testing Strategy

The pytest suite covers:

* feature engineering
* data validation
* training
* evaluation
* prediction
* serialization parity
* API behavior

The test suite uses:

* reusable pytest fixtures
* parametrized feature tests
* mocking
* prediction determinism checks
* serialization parity tests
* FastAPI `TestClient`

Coverage gate:

`>= 70%`

The handbook specifies these testing requirements and requires the coverage gate in `pyproject.toml`.

---

## 13. Dockerization

TODO — complete after the final Docker implementation.

### Single-stage image size

`TODO`

### Multi-stage image size

`TODO`

### Docker Hub image

`TODO`

The handbook requires a multi-stage image, a non-root runtime user, a health check, and publication to Docker Hub.

---

## 14. Maturity Self-Assessment

Current maturity level:

`TODO`

### What is missing to reach the next level?

`TODO`

The next module introduces experiment tracking, model registry, DVC-based data versioning, CI/CD, and automated retraining, so this is the capability gap between Module 1 and Module 2.

---

## 15. Definition of Done

* [ ] Public GitHub repository with meaningful commit history
* [ ] Package installs successfully in a clean environment
* [ ] Linting passes
* [ ] Pre-commit hooks installed
* [ ] Tests pass with coverage >= 70%
* [ ] `/health` works
* [ ] `/metadata` works
* [ ] `/predict` works
* [ ] `/predict/batch` works
* [ ] ONNX parity test passes
* [ ] Pickle vs ONNX latency numbers recorded
* [ ] Docker image is multi-stage
* [ ] Docker runtime uses a non-root user
* [ ] Docker image published to Docker Hub
* [ ] README provides a three-command quickstart
* [ ] Maturity self-assessment completed
* [ ] Pull request reviewed and merged
* [ ] `v0.1.0` release tagged
