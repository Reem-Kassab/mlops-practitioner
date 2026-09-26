from pydantic import BaseModel, Field
from typing import List

# 1. the data that the user should send
class TripInput(BaseModel):
    PULocationID: int = Field(..., description="Pickup Location ID")
    DOLocationID: int = Field(..., description="Dropoff Location ID")
    trip_distance: float = Field(..., gt=0.0, description="Trip distance in miles, must be greater than 0")

# 2. the result for one output
class PredictionOutput(BaseModel):
    duration_minutes: float

# 3. the result for one batch
class BatchTripInput(BaseModel):
    trips: List[TripInput]

# 4. the result for group of batchs
class BatchPredictionOutput(BaseModel):
    predictions: List[float]

# 5. healthy check response
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool