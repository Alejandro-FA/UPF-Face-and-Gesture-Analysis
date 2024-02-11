from .face_detector import FaceDetector, DetectionResult
from mtcnn import MTCNN
import imageio.v2
import cv2
import os
import contextlib

class MTCNNDetector(FaceDetector):
    def __init__(self) -> None:
        super().__init__()
        self.detector = MTCNN()

    def __call__(self, image: imageio.v2.Array) -> list[DetectionResult]:
        results = []
        # Convert the image from BGR to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Suppress output
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                detection_result = self.detector.detect_faces(img)

        for detection in detection_result:
            confidence = detection["confidence"]
            
            bbox = detection["box"]
            results.append(DetectionResult(confidence, *bbox))

        return results


    def save(file_path: str) -> None:
        return NotImplementedError()