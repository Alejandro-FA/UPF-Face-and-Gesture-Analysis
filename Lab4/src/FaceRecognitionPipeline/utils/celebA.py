from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms
import os
import torch
import numpy as np


class CelebA(Dataset):
    """
    Required class to load the CelebA dataset
    """
    def __init__(self, path: str, ids_file: str, transform: torchvision.transforms = None) -> None:
        
        self.path = path
        if not self.path.endswith("/"): self.path += "/"

        # Verify that the path is valid
        if os.path.isdir(path) == False:
            raise ValueError(f"Invalid directory {path}")
        
        self.ids_file = ids_file
        self.transform = transform
        self.labels: dict[str, int] = self.__load_ids()
        self.labels_tensors = torch.tensor(list(self.labels.values()))
        print(f"Created dataset with {len(self)} images and {len(self.get_unique_labels())} unique labels.")
    
    
    def __load_ids(self) -> dict[str, int]:
        """
        Loads the information from the ids file.
        A line has the following format:
            XXXXXX.jpg <id>
        where XXXXXX represents the image number, and <id> represents the id of the person in that image.
        """
        
        # print("Loading ids...")
        try:
            # Annotations file contains the annotations for both the training and testing dataset.
            file = open(self.ids_file, "r").read().strip()
        except:
            raise FileNotFoundError(f"File {self.ids_file} could not be found")

        labels = {}
        for line in file.splitlines():
            splited_line = line.split(" ")
            img_num = splited_line[0].split(".")[0]
            id = int(splited_line[1])
            labels[img_num] = id
        
        # print("Ids loaded!")
        return labels

    
    def get_unique_labels(self) -> set[int]:
        return set(self.labels.values())

    
    ################################ Necerssary functions for pytorch data loader ################################
    def __getitem__(self, index):
        key = list(self.labels.keys())[index]
        # image = read_image(f"{self.path}{key}.jpg", ImageReadMode.RGB)
        
        # if self.transform is not None:
        #     image = self.transform(image)
        image = torch.load(f"{self.path}{key}.pt")
        return image, self.labels_tensors[index]
    

    def __len__(self):
        return len(self.labels)

