from abc import ABC, abstractmethod
from .bounding_box import BoundingBox
import imageio.v2



class DetectionResult:
    def __init__(self, probability: float, x: int, y: int, w: int, h: int) -> None:
        self.probability: float = probability
        self.bounding_box: BoundingBox = BoundingBox(x, y, w, h)
    
    def __str__(self) -> str:
        box = self.bounding_box
        str = f"""
        Probability: {self.probability}
        Bounding box: ({box.x1}, {box.y1}), ({box.width}, {box.height})
        """
        return str


class FaceDetector(ABC):
    def __call__(self, image: imageio.v2.Array) -> list[DetectionResult]:
        det_results = self.__detect_faces(image)
        return self.__get_largest_images(det_results)
    
    @abstractmethod
    def save(file_path: str) -> None:
        raise NotImplementedError("Implement in the subclass.")

    @abstractmethod
    def __detect_faces(self, det_results: list[DetectionResult], min_probability: float) -> list[DetectionResult]:
        raise NotImplementedError("Implement in the subclass.")


    def __get_largest_images(self, det_results: list[DetectionResult]) -> list[DetectionResult]:
        return sorted(det_results, key=lambda res: res.bounding_box.get_area(), reverse=True)





