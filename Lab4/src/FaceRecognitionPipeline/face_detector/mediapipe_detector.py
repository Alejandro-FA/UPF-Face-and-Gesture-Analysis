from .face_detector import FaceDetector, DetectionResult, BoundingBox
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import imageio.v2
from pathlib import Path
import numpy as np



class MediaPipeDetector(FaceDetector):
    def __init__(self, model_asset_path: str) -> None:
        super().__init__()
        model_path = Path(model_asset_path)
        assert model_path.exists(), "Model path does not exist"
        assert model_path.is_file(), "Model path is not a file"
        assert model_path.suffix == ".tflite", "Model path is not a .tflite file"

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
        self.detector = vision.FaceDetector.create_from_options(options)



    def detect_faces(self, image: imageio.v2.Array) -> list[DetectionResult]:
        results = []
        # https://developers.google.com/mediapipe/api/solutions/python/mp/Image
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))
        detection_result = self.detector.detect(rgb_frame)
        
        for detection in detection_result.detections:
            category = detection.categories[0]
            probability = round(category.score, 2)

            # Draw bounding_box
            bbox = detection.bounding_box
            bbox = BoundingBox(
                int(bbox.origin_x),
                int(bbox.origin_y),
                int(bbox.width),
                int(bbox.height),
            )
            results.append(DetectionResult(probability, bbox))
            
        return results
    

    def save(file_path: str) -> None:
        return NotImplementedError()


