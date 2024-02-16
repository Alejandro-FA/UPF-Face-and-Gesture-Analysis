from typing import Any
import imageio.v2
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
from ..utils import BoundingBox


class FeatureExtractorPreprocessor:
    def __init__(self, new_size: int, grayscale: bool = False) -> None:
        self.output_width = new_size
        self.output_height = new_size
        self.grayscale = grayscale
        self.torch_transform = transforms.Compose([transforms.ToTensor()])
    

    def __call__(self, image: imageio.v2.Array, bbox: BoundingBox) -> torch.Tensor:
        image = self.__change_color(image)
        image = self.__square_crop(image, bbox)
        
        if image.shape[0] > self.output_width:
            image = self.__downscale(image) # Output image should be downscaled
        else:
            image = self.__upscale(image) # Output image should be upscaled
        
        assert image.shape[0] == self.output_width, "Image width is not the expected"
        assert image.shape[1] == self.output_height, "Image height is not the expected"
        return self.__to_torch(image)
            

    def __to_torch(self, image: imageio.v2.Array) -> torch.Tensor:
        """
        Change the image format to the one used in the rest of the project.
        """
        return self.torch_transform(image)
    

    def __square_crop(self, image: imageio.v2.Array, bbox: BoundingBox) -> imageio.v2.Array:
        """
        Crop the image to a square around the bounding box.
        """
        assert bbox.x1 >= 0, f"{bbox.x1} < 0"
        assert bbox.y1 >= 0, f"{bbox.y1} < 0"
        assert bbox.x2 < image.shape[1], f"{bbox.x2} >= {image.shape[1]}"
        assert bbox.y2 < image.shape[0], f"{bbox.y2} >= {image.shape[0]}"
        assert bbox.width > 0, "width <= 0"
        assert bbox.height > 0, "height <= 0"

        bbox = bbox.copy()
        if bbox.width == bbox.height:
            return image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        
        # Calculate the margin to add to the image
        x_margin = 0 if bbox.width > bbox.height else bbox.height - bbox.width
        y_margin = 0 if bbox.height > bbox.width else bbox.width - bbox.height

        # Delete 1 pixel from the bounding box if the margin is odd
        if x_margin % 2 != 0:
            bbox.height -= 1
        if y_margin % 2 != 0:
            bbox.width -= 1
        
        x_margin = x_margin // 2
        y_margin = y_margin // 2

        # Check if the margin goes out of the image
        out_of_bound_pixels = abs(min(0, bbox.x1 - x_margin))
        out_of_bound_pixels = max(out_of_bound_pixels, abs(min(0, bbox.y1 - y_margin)))
        out_of_bound_pixels = max(out_of_bound_pixels, abs(max(image.shape[1] - 1, bbox.x2 + x_margin) - image.shape[1] + 1))
        out_of_bound_pixels = max(out_of_bound_pixels, abs(max(image.shape[0] - 1, bbox.y2 + y_margin) - image.shape[0] + 1))
        
        x_margin -= out_of_bound_pixels
        y_margin -= out_of_bound_pixels
        
        assert bbox.x1 - x_margin >= 0, "x1 - x_margin < 0"
        assert bbox.y1 - y_margin >= 0, "y1 - y_margin < 0"
        assert bbox.x2 + x_margin < image.shape[1], "x2 + x_margin >= image.shape[1]"
        assert bbox.y2 + y_margin < image.shape[0], "y2 + y_margin >= image.shape[0]"
        image = image[bbox.y1 - y_margin:bbox.y2 + y_margin, bbox.x1 - x_margin:bbox.x2 + x_margin]
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

    
    def __change_color(self, image: imageio.v2.Array) -> imageio.v2.Array:
        """
        Change the color format of the image.
        """
        if self.grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if not self.grayscale and len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image