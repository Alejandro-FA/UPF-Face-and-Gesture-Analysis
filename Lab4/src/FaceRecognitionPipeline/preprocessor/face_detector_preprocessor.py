import imageio.v2
import cv2
import numpy as np
from typing import Literal

class FaceDetectorPreprocessor:
    def __init__(self, output_channels: Literal[1, 3] = 3) -> None:
        self.output_channels = output_channels


    def __call__(self, image: imageio.v2.Array) -> imageio.v2.Array:
        image = self.__to_3_dims(image)
        image = self.__change_channels(image)
        return image.astype(np.uint8)
    

    def __to_3_dims(self, image: imageio.v2.Array) -> imageio.v2.Array:
        """
        Convert the image to 3 dimensions if it is not.
        """
        if len(image.shape) == 3:
            return image
        if len(image.shape) == 2:
            return np.expand_dims(image, axis=2)
        else:
            raise ValueError("Image must have either 2 or 3 dimensions!")
    

    def __change_channels(self, image: imageio.v2.Array) -> imageio.v2.Array:
        """
        Change the number of channels of the image.
        """
        channels = image.shape[2]
        if self.output_channels == 1 and channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif self.output_channels == 3 and channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    
    