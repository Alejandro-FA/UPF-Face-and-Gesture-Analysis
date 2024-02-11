from typing import Any
from face_detector import FaceDetector
from feature_extractor import FeatureExtractor

class FR_Model:
    def __init__(self, face_detector_path: str, feature_extractor_path: str):
        # self.face_detector: FaceDetector = subclass of FaceDetector
        # self.feature_extractor: FeatureExtractor = subclass of FeatureExtractor
        pass
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass