import datetime
import numpy as np
import os
from typing import Literal
from tqdm import tqdm
from imageio.v2 import imread
from torchvision import transforms


def get_log_path(base_path: str, extension: str = "log") -> str:
    """
    Generates a log file path based on the given base path and extension.

    Args:
        base_path (str): The base path where the log file will be saved.
        extension (str, optional): The file extension for the log file. Defaults to "log".

    Returns:
        str: The generated log file path.
    """
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H:%M")
    return f"{base_path}_{now_str}.{extension}"



def load_images(images_paths: list[str]) -> np.ndarray:
    images = []
    try:
        # Store all images in memory
        for image_path in tqdm(images_paths, desc=f"Loading images"):
            images.append(imread(image_path))
    except FileNotFoundError as e:
        print(f"Error when opening image {e.filename}. File not found.")
        raise e
    
    return np.array(images)



def get_images_paths(images_dir: str, input_format: Literal["jpg", "png", "pt"]):
    dir_elements = os.listdir(images_dir)
    dir_files = [os.path.join(images_dir, f) for f in dir_elements if os.path.isfile(os.path.join(images_dir, f))]
    return [f for f in dir_files if f.endswith(f".{input_format}")]