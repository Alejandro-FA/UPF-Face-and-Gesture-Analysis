import os
import shutil
import numpy as np
from abc import ABC, abstractmethod

class DatasetSplitter(ABC):
    def __init__(self, cropped_imgs_path: str, target_dataset_path: str, annotations_path: str) -> None:
        if self.__valid_dir_path(cropped_imgs_path) == False:
            raise ValueError(f"Path {cropped_imgs_path} does not exist.")
        else:
            self.cropped_images_path = cropped_imgs_path
            self.cropped_images_path = self.__remove_backslash(cropped_imgs_path)
        
        if self.__valid_dir_path(target_dataset_path) == False:
            print(f"Target directory {target_dataset_path} does not exist. Creating it...")
            os.makedirs(target_dataset_path)
        else:
            print(f"[WARNING] Directory {target_dataset_path} already exists. All of its contents will be modified.")
            shutil.rmtree(target_dataset_path)
            os.makedirs(target_dataset_path)
        
        self.target_dataset_path = self.__remove_backslash(target_dataset_path)
        self.annotations_path = annotations_path
        
        self.ids_count = self.__get_ids_count(cropped_imgs_path)
        self.total_images = np.sum(list(self.ids_count.values()))
        print(f"Total images: {self.total_images}")
    
    
    @abstractmethod
    def from_cropped_to_dataset(self) -> dict[int, int]:
        raise NotImplementedError("Implement in the subclass.")
    
        
    def __valid_dir_path(self, path: str) -> bool:
        return os.path.isdir(path)
    
    def __remove_backslash(self, path: str):
        if path.endswith("/"): return path[0:-1]
        else: return path
        
    def generate_image_name(self, image_num):
        image_num = str(image_num)
        remaining_length = len(str(self.total_images)) - len(image_num)
        
        return "0" * remaining_length + image_num + ".jpg"

    
    def __get_ids_count(self, path: str):
        ids_count = {}
        for directory in sorted(os.listdir(path)):
            curr_id = int(directory[1:])
            ids_count[curr_id] = 0
            for file in os.listdir(f"{path}/{directory}"):
                ids_count[curr_id] += 1
                    
        return ids_count