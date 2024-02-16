from .face_detector import FaceDetector, DetectionResult
import imageio.v2
import contextlib
import os
import tensorflow as tf
from mtcnn import MTCNN


class MTCNNDetector(FaceDetector):
    def __init__(self) -> None:
        super().__init__()
        self.detector = MTCNN()

    def detect_faces(self, image: imageio.v2.Array) -> list[DetectionResult]:
        results = []

        # Suppress output
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                with tf.device('/CPU:0'):
                    detection_result = self.detector.detect_faces(image)

        for detection in detection_result:
            confidence = detection["confidence"]
            
            bbox = detection["box"]
            results.append(DetectionResult(confidence, *bbox))

        return results


    def save(file_path: str) -> None:
        return NotImplementedError()