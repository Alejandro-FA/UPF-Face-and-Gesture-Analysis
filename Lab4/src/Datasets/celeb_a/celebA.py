from torch.utils.data import Dataset
import os
import torch
from .utils import get_ids
from PIL import Image
from torchvision import transforms
from ..utils import get_images_paths


class CelebA(Dataset):
    """
    Required class to load the CelebA dataset
    """
    def __init__(self, images_dir: str, ids_file_path: str) -> None:
        if not os.path.isdir(images_dir):
            raise ValueError(f"Invalid directory {images_dir}")
        if not os.path.isfile(ids_file_path):
            raise ValueError(f"Invalid file {ids_file_path}")
        
        self.transform = transforms.ToTensor()
        ids: dict[str, int] = get_ids(ids_file_path, extension="jpg")
        self.images_paths = get_images_paths(images_dir, input_format="jpg")
        self.labels = {os.path.join(images_dir, k): torch.tensor(v) for k, v in ids.items()}

        print(f"Created dataset with {len(self)} images and {self.num_unique_labels()} unique labels.")


    def num_unique_labels(self) -> int:
        labels = {tensor.item() for tensor in self.labels.values()}
        return len(labels)


    ################################ Necerssary functions for pytorch data loader ################################
    def __getitem__(self, index):
        file_path = self.images_paths[index]
        image = self.transform(Image.open(file_path))
        return image, self.labels[file_path]
    

    def __len__(self):
        return len(self.images_paths)
    