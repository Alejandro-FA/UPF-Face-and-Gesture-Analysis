import os, shutil
import numpy as np
import sys
from .splitter import DatasetSplitter


class VGGFace2Splitter(DatasetSplitter):
    
    
    def __init__(self, cropped_imgs_path: str, target_dataset_path: str, annotations_path: str) -> None:
        super().__init__(cropped_imgs_path, target_dataset_path, annotations_path)
    
    def from_cropped_to_dataset(self) -> dict[int, int]:
        """
        This method merges all the images present in the cropped_images_path into the folder specified by target_dataset_path.
        While doing so, it changes the names of the images and generates the labels path.
        
        Returns:
            A dictionary containing the number of images for each id
        """
        ids_count = {}
        total_images = 0
        annotations_file = open(f"{self.annotations_path}", "w")
        
        for directory in os.listdir(self.cropped_images_path):
            curr_id = int(directory[1:]) # Format of the folder: nXXXXXX
            complete_path = self.cropped_images_path + f"/{directory}"
            ids_count[curr_id] = 0
            
            for file in os.listdir(complete_path):
                total_images += 1
                image_name = super().generate_image_name(total_images)
                
                # Copy the image to the destination path with its new name
                shutil.copy(f"{complete_path}/{file}", f"{self.target_dataset_path}/{image_name}")
                
                # Write the label of the new image to the labels path
                annotations_file.write(f"{image_name} {curr_id}\n")
                
                ids_count[curr_id] += 1
        
        annotations_file.close()
        
        return ids_count
    