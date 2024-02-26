import imageio.v2
import cv2
from ..face_detector import BoundingBox
from typing import Literal
import numpy as np


class FeatureExtractorPreprocessor:
    def __init__(self, new_size: int, output_channels: Literal[1, 3] = 3, color_transform: int = None) -> None:
        self.output_width = new_size
        self.output_height = new_size
        self.output_channels = output_channels
        self.color_transform = color_transform
    

    def __call__(self, image: imageio.v2.Array, bbox: BoundingBox) -> imageio.v2.Array:
        image = self.__to_3_dims(image)
        image = self.__change_channels(image)
        
        if self.color_transform is not None:
            image = cv2.cvtColor(image, self.color_transform)
        
        image = self.__square_crop(image, bbox)
        
        if image.shape[0] > self.output_width:
            image = self.__downscale(image) # Output image should be downscaled
        elif image.shape[0] < self.output_width:
            image = self.__upscale(image) # Output image should be upscaled
        
        
        assert image.shape[0] == self.output_width, "Image width is not the expected"
        assert image.shape[1] == self.output_height, "Image height is not the expected"
        return image
    

    def __square_crop(self, image: imageio.v2.Array, bbox: BoundingBox) -> imageio.v2.Array:
        """
        Crop the image to a square around the bounding box. We avoid resizing
        the image to avoid losing information / distorting the image.
        """
        if bbox.width == bbox.height:
            return image[bbox.y0:bbox.y1, bbox.x0:bbox.x1]
        
        # Calculate the margin to add to the image
        x_margin = 0 if bbox.width > bbox.height else bbox.height - bbox.width
        y_margin = 0 if bbox.height > bbox.width else bbox.width - bbox.height

        # Delete 1 pixel from the bounding box if the margin is odd
        if x_margin % 2 != 0:
            bbox.height -= 1
        if y_margin % 2 != 0:
            bbox.width -= 1
        
        # Split margin equally in both sides
        x_margin = x_margin // 2
        y_margin = y_margin // 2

        # Reduce margin if the square crop is out of the image
        out_of_bound_pixels = abs(min(0, bbox.x0 - x_margin))
        out_of_bound_pixels = max(out_of_bound_pixels, abs(min(0, bbox.y0 - y_margin)))
        out_of_bound_pixels = max(out_of_bound_pixels, abs(max(image.shape[1] - 1, bbox.x1 + x_margin) - image.shape[1] + 1))
        out_of_bound_pixels = max(out_of_bound_pixels, abs(max(image.shape[0] - 1, bbox.y1 + y_margin) - image.shape[0] + 1))
        
        x_margin -= out_of_bound_pixels
        y_margin -= out_of_bound_pixels
        
        # Return the square crop around the bounding box
        assert bbox.x0 - x_margin >= 0, "x1 - x_margin < 0"
        assert bbox.y0 - y_margin >= 0, "y1 - y_margin < 0"
        assert bbox.x1 + x_margin < image.shape[1], "x2 + x_margin >= image.shape[1]"
        assert bbox.y1 + y_margin < image.shape[0], "y2 + y_margin >= image.shape[0]"
        image = image[bbox.y0 - y_margin:bbox.y1 + y_margin, bbox.x0 - x_margin:bbox.x1 + x_margin]
        assert image.shape[0] == image.shape[1], f"Image of shape ({image.shape[1]}, {image.shape[0]}) is not square"
        return image


    def __downscale(self, image: imageio.v2.Array) -> imageio.v2.Array:
        """
        Downscale the image to a size of width x height.
        """
        return cv2.resize(image, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
        
    
    def __upscale(self, image: imageio.v2.Array) -> imageio.v2.Array:
        """
        Downscale the image to a size of width x height.
        """
        return cv2.resize(image, (self.output_width, self.output_height), interpolation=cv2.INTER_CUBIC)

    
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
    