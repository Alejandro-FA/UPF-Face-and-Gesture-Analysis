import imageio.v2
import cv2
import numpy as np

class FaceDetectorPreprocessor:
    def __init__(self, grayscale: bool = False) -> None:
        self.grayscale = grayscale

    def __call__(self, image: imageio.v2.Array) -> imageio.v2.Array:
        if self.grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if not self.grayscale and len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        return image.astype(np.uint8)
    