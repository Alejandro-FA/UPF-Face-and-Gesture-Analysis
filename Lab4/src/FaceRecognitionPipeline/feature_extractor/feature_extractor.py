from abc import ABC, abstractmethod
import torch

class FeatureExtractor(ABC):
    @abstractmethod
    def __call__(self, image: torch.Tensor) -> tuple[int, float]:
        raise NotImplementedError("Implement in the subclass.")

    @abstractmethod
    def save(file_path: str) -> None:
        raise NotImplementedError("Implement in the subclass.")
    
    def num_parameters(self):
        raise NotImplementedError("Implemenet in the subclass")