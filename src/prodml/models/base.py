from abc import ABC, abstractmethod
from typing import Any


class ModelBase(ABC):
    """Abstract interface for model implementations."""

    @abstractmethod
    def load(self) -> None:
        """Load the model artifact into memory."""
        raise NotImplementedError

    @abstractmethod
    def predict_one(self, features: dict[str, Any]) -> float:
        """Generate one prediction."""
        raise NotImplementedError

    @abstractmethod
    def predict_batch(
        self,
        features: list[dict[str, Any]],
    ) -> list[float]:
        """Generate predictions for multiple inputs."""
        raise NotImplementedError