from utils import BoundingBox
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class DNNDetector:
    def __init__(self) -> None:
        base_options = python.BaseOptions(model_asset_path='mediapipe/detector.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)


    def detect_faces(self, image_path: str) -> list[tuple[BoundingBox, float]]:
        results = []
        image = mp.Image.create_from_file(image_path)
        detection_result = self.detector.detect(image)
        
        for detection in detection_result.detections:
            # Decide whether it is a human face or not depending on the output probability
            category = detection.categories[0]
            probability = round(category.score, 2)

            # Draw bounding_box
            # if probability > self.threshold:
            bbox = detection.bounding_box
            results.append((BoundingBox(
                bbox.origin_x,
                bbox.origin_y,
                bbox.width,
                bbox.height,
            ), probability))
            
        return results