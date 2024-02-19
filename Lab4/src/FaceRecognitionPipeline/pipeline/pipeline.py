from ..face_detector import FaceDetector, DetectionResult, BoundingBox
from ..preprocessor import FaceDetectorPreprocessor, FeatureExtractorPreprocessor
from ..feature_extractor import FeatureExtractor
import imageio.v2


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
        detection_results: list[DetectionResult] = self.face_detector(preprocessed_image)   
        
        for res in detection_results:
            probability: float = res.probability # TODO: Decide whether to use this probability or not
            bounding_box: BoundingBox = res.bounding_box

            boxed_image: imageio.v2.Array = self.fe_preprocessor(preprocessed_image, bounding_box)
            id: int = self.feature_extractor(boxed_image)
            if id != -1:
                # There are no images with more than one user in them
                return id
        
        return -1
        
        