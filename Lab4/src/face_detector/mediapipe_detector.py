from .face_detector import FaceDetector, DetectionResult
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import imageio.v2
import cv2
from pathlib import Path



class MediaPipeDetector(FaceDetector):
    def __init__(self, model_path: Path) -> None:
        super().__init__()
        assert model_path.exists(), "Model path does not exist"
        assert model_path.is_file(), "Model path is not a file"
        assert model_path.suffix == ".tflite", "Model path is not a .tflite file"

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)


    def __call__(self, image: imageio.v2.Array) -> list[DetectionResult]:
        results = []
        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection_result = self.detector.detect(image)
        
        for detection in detection_result.detections:
            # Decide whether it is a human face or not depending on the output probability
            category = detection.categories[0]
            probability = round(category.score, 2)

            # Draw bounding_box
            bbox = detection.bounding_box
            results.append(
                DetectionResult(
                    probability,
                    bbox.origin_x,
                    bbox.origin_y,
                    bbox.width,
                    bbox.height,
                )
            )
            
        return results
    

    def save(file_path: str) -> None:
        return NotImplementedError()


