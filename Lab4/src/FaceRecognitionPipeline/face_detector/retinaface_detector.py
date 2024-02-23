from .face_detector import FaceDetector, DetectionResult, BoundingBox
import imageio.v2
from retinaface import RetinaFace
from typing import Any


class RetinaFaceDetector(FaceDetector):
    def __init__(self, threshold: float=0.5) -> None:
        super().__init__()
        self.threshold = threshold


    # NOTE: The RetinaFace library has another method called `extract_faces`
    # that directly returns the detected faces. It might be worth exploring
    # because it also allows to align the faces.
    def detect_faces(self, images: list[imageio.v2.Array]) -> list[list[DetectionResult]]:
        global_results = []

        for im in images:
            faces = RetinaFace.detect_faces(im, threshold=self.threshold)
            results = self.__get_detection_results(faces)
            global_results.append(results)

        return global_results
    
    
    def __get_detection_results(self, faces: dict[str, dict[str, Any]]) -> list[DetectionResult]:
        results = []
        for face in faces.values():
            facial_area = face['facial_area']
            bbox = BoundingBox.from_coords(
                x0=facial_area[0],
                y0=facial_area[1],
                x1=facial_area[2],
                y1=facial_area[3]
            )
            prob = face['score']
            results.append(DetectionResult(prob, bbox))
        
        return results
