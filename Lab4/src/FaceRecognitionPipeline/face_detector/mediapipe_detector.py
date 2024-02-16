from .face_detector import FaceDetector, DetectionResult
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import imageio.v2
import cv2
from pathlib import Path
import numpy as np



class MediaPipeDetector(FaceDetector):
    def __init__(self, file_name: str) -> None:
        super().__init__()
        model_path = Path(file_name)
        assert model_path.exists(), "Model path does not exist"
        assert model_path.is_file(), "Model path is not a file"
        assert model_path.suffix == ".tflite", "Model path is not a .tflite file"

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)



    def detect_faces(self, image: imageio.v2.Array) -> list[DetectionResult]:
        results = []
        # Convert the image from BGR to RGB
        # https://developers.google.com/mediapipe/api/solutions/python/mp/Image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))

        # image = np.asarray(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection_result = self.detector.detect(rgb_frame)
        
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


