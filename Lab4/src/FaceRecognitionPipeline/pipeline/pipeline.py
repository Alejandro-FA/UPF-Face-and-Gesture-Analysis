from ..face_detector import FaceDetector
from ..preprocessor import FaceDetectorPreprocessor, FeatureExtractorPreprocessor
from ..feature_extractor import FeatureExtractor
import imageio.v2
from typing import Any

class Pipeline:
    def __init__(
        self,
        fd_preprocessor: FaceDetectorPreprocessor,
        face_detector: FaceDetector,
        fe_preprocessor: FeatureExtractorPreprocessor,
        feature_extractor: FeatureExtractor
    ) -> None:
        
        self.fd_preprocessor = fd_preprocessor
        self.face_detector = face_detector
        self.fe_preprocessor = fe_preprocessor
        self.feature_extractor = feature_extractor

    def __call__(self, image: imageio.v2.Array) -> Any:
        preprocessed_image = self.fd_preprocessor(image)
        detection_results = self.face_detector(image)   
        
        for res in detection_results:
            face_image = self.fe_preprocessor(preprocessed_image, res.bbox)
            id = self.feature_extractor(face_image)
            if id != -1:
                # There are no images with more than one user in them
                return id
        
        return -1
        
        