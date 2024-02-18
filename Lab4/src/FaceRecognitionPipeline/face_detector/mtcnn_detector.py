from .face_detector import FaceDetector, DetectionResult, BoundingBox
import imageio.v2
from facenet_pytorch import MTCNN
import MyTorchWrapper as mtw
import numpy as np


class MTCNNDetector(FaceDetector):
    def __init__(self, use_gpu: bool=False, thresholds: tuple[float, float, float]=[0.6, 0.7, 0.7]) -> None:
        super().__init__()
        device = mtw.get_torch_device(use_gpu=use_gpu, debug=False)
        self.mtcnn = MTCNN(post_process=False, keep_all=True, device=device, thresholds=thresholds)

    def detect_faces(self, images: list[imageio.v2.Array]) -> list[list[DetectionResult]]:
        global_results = []
        boxes, probs = self.mtcnn.detect(images, landmarks=False) # Process multiple images at once

        for i in range(len(images)):
            results = self.__get_detection_results(boxes[i], probs[i])
            global_results.append(results)

        return global_results


    def save(file_path: str) -> None:
        return NotImplementedError()
    

    def __get_detection_results(self, boxes: np.ndarray, probs: list[float]) -> list[DetectionResult]:
        if boxes is None or probs is None:
            return []

        results = []
        for bbox, prob in zip(boxes, probs):
            bbox = BoundingBox.from_coords(*bbox.astype(int))
            results.append(DetectionResult(prob, bbox))
        
        return results
