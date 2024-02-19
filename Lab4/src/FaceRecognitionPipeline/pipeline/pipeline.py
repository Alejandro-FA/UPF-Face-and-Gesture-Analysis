from ..face_detector import FaceDetector, DetectionResult, BoundingBox
from ..preprocessor import FaceDetectorPreprocessor, FeatureExtractorPreprocessor
from ..feature_extractor import FeatureExtractor
import imageio.v2
import numpy as np


class Pipeline:
    def __init__(
        self,
        fd_preprocessor: FaceDetectorPreprocessor,
        face_detector: FaceDetector,
        fe_preprocessor: FeatureExtractorPreprocessor,
        feature_extractor: FeatureExtractor,
        detection_min_prob: float = 0.3,
        classification_min_prob: float = 0.3,
    ) -> None:
        self.fd_preprocessor = fd_preprocessor
        self.face_detector = face_detector
        self.fe_preprocessor = fe_preprocessor
        self.feature_extractor = feature_extractor
        self.detection_min_prob = detection_min_prob
        self.classification_min_prob = classification_min_prob


    def __call__(self, image: imageio.v2.Array) -> int:
        preprocessed_image: imageio.v2.Array = self.fd_preprocessor(image)
        detection_results: list[DetectionResult] = self.face_detector(preprocessed_image)   
        
        ids = [-1] # -1 is the id for no detection
        classification_probs = [0.0]
        for res in detection_results:
            bounding_box: BoundingBox = res.bounding_box
            detection_prob: float = res.probability

            # If the detection probability is lower than the threshold, we skip
            # the classification step
            if detection_prob >= self.detection_min_prob:
                boxed_image: imageio.v2.Array = self.fe_preprocessor(preprocessed_image, bounding_box)
                id, classification_prob = self.feature_extractor(boxed_image)
                ids.append(id)
                classification_probs.append(classification_prob)

        # Return the id with the highest classification probability or -1 if no ids
        best_prob_idx = np.argmax(classification_probs)
        best_id = ids[best_prob_idx]
        return best_id if classification_probs[best_prob_idx] >= self.classification_min_prob else -1
        
        