from ..face_detector import FaceDetector, DetectionResult
from ..preprocessor import FaceDetectorPreprocessor, FeatureExtractorPreprocessor
from ..feature_extractor import FeatureExtractor
import imageio.v2
import torch

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

    def __call__(self, image: imageio.v2.Array) -> int:
        preprocessed_image: imageio.v2.Array = self.fd_preprocessor(image)
        detection_results: list[DetectionResult] = self.face_detector(image)   
        
        for res in detection_results:
            face_image: torch.Tensor = self.fe_preprocessor(preprocessed_image, res.bbox)
            id: int = self.feature_extractor(face_image)
            if id != -1:
                # There are no images with more than one user in them
                return id
        
        return -1
        
        