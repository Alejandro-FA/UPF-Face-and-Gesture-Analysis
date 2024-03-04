from ..face_detector import FaceDetector, DetectionResult, BoundingBox
from ..preprocessor import FaceDetectorPreprocessor, FeatureExtractorPreprocessor
from ..feature_extractor import FeatureExtractor
import imageio.v2
import numpy as np
import os
import cv2


class Pipeline:
    def __init__(
        self,
        fd_preprocessor: FaceDetectorPreprocessor,
        face_detector: FaceDetector,
        fe_preprocessor: FeatureExtractorPreprocessor,
        feature_extractor: FeatureExtractor,
        detection_min_prob: float = 0.5,
        classification_min_prob: float = 0.2,
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
    
    
    def visualize(self, image: imageio.v2.Array, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the original image
        imageio.imwrite(f"{output_dir}/original_image.png", image)
        
        # Save the Face Detector preprocessed image
        fd_preprocessed = self.fd_preprocessor(image)
        imageio.imwrite(f"{output_dir}/fd_preprocessed.png", fd_preprocessed)
        
        # Save the image with the bounding box
        results: list[DetectionResult] = self.face_detector(fd_preprocessed)
        bbox_image = cv2.cvtColor(fd_preprocessed, cv2.COLOR_RGB2BGR)
        bounding_box: BoundingBox = results[0].bounding_box
        x0 = bounding_box.x0
        y0 = bounding_box.y0
        x1 = bounding_box.x1
        y1 = bounding_box.y1
        curr_image = cv2.rectangle(bbox_image, (x0, y0), (x1, y1), (0, 255, 0), 2)        
        cv2.imwrite(f"{output_dir}/bbox_image.png", curr_image)
        
        # Save the Feature Extractor preprocessed image
        boxed_image: imageio.v2.Array = self.fe_preprocessor(fd_preprocessed, bounding_box)
        boxed_image = cv2.cvtColor(boxed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_dir}/fe_preprocessed.png", boxed_image)
        
        
    
        self.feature_extractor.visualize(boxed_image, output_dir)
        
        
        
        
        