from torch.utils.data import Dataset
import os
import torch
from .utils import get_ids
from typing import Literal
from PIL import Image
from torchvision import transforms


class CelebA(Dataset):
    """
    Required class to load the CelebA dataset
    """
    def __init__(self, path: str, ids_file_path: str, input_format: Literal["jpg", "png", "pt"] = "pt") -> None:
        ids: dict[str, int] = get_ids(ids_file_path, extension=input_format)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.input_format = input_format
        self.image_paths = CelebA.__get_images_paths(path, input_format)
        self.labels = {os.path.join(path, k): torch.tensor(v) for k, v in ids.items()}
        print(f"Created dataset with {len(self)} images and {self.num_unique_labels()} unique labels.")

    def num_unique_labels(self) -> int:
        labels = {tensor.item() for tensor in self.labels.values()}
        return len(labels)

    @staticmethod
    def __get_images_paths(path: str, input_format: Literal["jpg", "png", "pt"]):
        if not os.path.isdir(path):
            raise ValueError(f"Invalid directory {path}")
        dir_elements = os.listdir(path)
        dir_files = [os.path.join(path, f) for f in dir_elements if os.path.isfile(os.path.join(path, f))]
        return [f for f in dir_files if f.endswith(f".{input_format}")]

    
    ################################ Necerssary functions for pytorch data loader ################################
    def __getitem__(self, index):
        file_path = self.image_paths[index]
    
        if self.input_format == "pt":
            image = torch.load(file_path)
        else:
            image = Image.open(file_path)
            image = self.transform(image)
        return image, self.labels[file_path]
    

    def __len__(self):
        return len(self.image_paths)
    

