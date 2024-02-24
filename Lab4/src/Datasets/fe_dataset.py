from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
from .utils import get_ids, get_images_paths, get_num_unique_ids
from PIL import Image


class FeatureExtractorDataset(Dataset):
    """
    Required class to load the any of the datasets used to train a feature extractor dataset provided for training
    """
    def __init__(self, images_dir: str, ids_file_path: str) -> None:
        if not os.path.isdir(images_dir):
            raise ValueError(f"Invalid directory {images_dir}")
        if not os.path.isfile(ids_file_path):
            raise ValueError(f"Invalid file {ids_file_path}")
    
        self.transform = transforms.ToTensor()
        ids: dict[str, int] = get_ids(ids_file_path, extension="jpg")
        self.num_classes = get_num_unique_ids(ids_file_path)
        self.images_paths = get_images_paths(images_dir, input_format="jpg")
        self.labels = {os.path.join(images_dir, k): torch.tensor(v) for k, v in ids.items()}

        print(f"Created dataset with {len(self)} images and {self.num_classes} unique labels.")

    
    ################################ Necerssary functions for pytorch data loader ################################
    def __getitem__(self, index):
        file_path = self.images_paths[index]
        image = self.transform(Image.open(file_path))
        return image, self.labels[file_path]
    

    def __len__(self):
        return len(self.images_paths)