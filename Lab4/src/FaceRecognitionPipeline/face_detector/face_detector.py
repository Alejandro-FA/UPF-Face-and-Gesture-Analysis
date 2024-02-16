from abc import ABC, abstractmethod
import imageio.v2
from ..utils import BoundingBox



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
        det_results = self.detect_faces(image)
        for res in det_results:
            res.bounding_box.fit_to_image(image)
        return self.__get_largest_images(det_results, 2)
    
    @abstractmethod
    def save(file_path: str) -> None:
        raise NotImplementedError("Implement in the subclass.")

    @abstractmethod
    def detect_faces(self, image: imageio.v2.Array) -> list[DetectionResult]:
        raise NotImplementedError("Implement in the subclass.")


    def __get_largest_images(self, det_results: list[DetectionResult], n: int) -> list[DetectionResult]:
        return sorted(det_results, key=lambda res: res.bounding_box.get_area(), reverse=True)[0:n]





