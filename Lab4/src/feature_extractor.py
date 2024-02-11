from typing import Any
from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("Implement in the subclass.")

    @abstractmethod
    def save(file_path: str) -> None:
        raise NotImplementedError("Implement in the subclass.")