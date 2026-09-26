# Taxi Trip Duration Predictor

## What is Taxi Trip Duration Predictor?
This is a production-ready Machine Learning REST API that predicts the duration of NYC taxi trips in minutes based on the pickup location, dropoff location, and trip distance. The service is built with FastAPI, uses an ONNX-optimized model for low-latency inference, and is fully containerized using a multi-stage Docker build.

## How do I run it?
You can go from zero to a prediction in just three commands using Docker. No local Python environment is required.

**1. Build the Docker image:**
```bash
docker build -t taxi-duration-api .
```
**2. Run the container:**

``` Bash
docker run -p 8000:8000 taxi-duration-api
```
**3. Make a prediction:**

```Bash
curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
     -H "Content-Type: application/json" \
     -d '{"PULocationID": 236, "DOLocationID": 239, "trip_distance": 2.5}'
```
## How do I call /predict?
Send a POST request with the single trip details.

```Bash
curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
     -H "Content-Type: application/json" \
     -d '{
           "PULocationID": 236,
           "DOLocationID": 239,
           "trip_distance": 2.5
         }'
```
## How do I call /predict/batch?
Send a POST request with an array of trips.

```Bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"PULocationID": 236, "DOLocationID": 239, "trip_distance": 2.5}'
```
## What does the repository contain?
```text
.
├── Dockerfile                      # Multi-stage Docker build configuration
├── pyproject.toml                  # Python package and dependency declarations
├── uv.lock                         # Deterministic dependency lockfile
├── README.md                       # Project documentation
├── reports/
│   └── module-1.md                 # Engineering metrics, latency comparisons, and maturity assessment
├── models/
│   ├── baseline_linear_pipeline.pkl  # Original scikit-learn pipeline
│   └── baseline_linear_pipeline.onnx # Optimized ONNX model for production inference
├── tests/
│   └── test_api.py                 # Pytest suite for FastAPI endpoints
└── src/
    └── taxi_duration/
        ├── config.py               # Pydantic environment configurations
        ├── data.py                 # Data ingestion and cleaning
        ├── features.py             # Feature engineering logic
        ├── logging_conf.py         # Custom JSON logger configuration
        ├── train.py                # Model training pipeline orchestration
        ├── export.py               # Script to convert PKL to ONNX and run parity tests
        ├── predict.py              # Core inference class managing the ONNX session
        └── api/
            ├── schemas.py          # Pydantic validation schemas for API requests
            └── main.py             # FastAPI application and endpoint definitions
```