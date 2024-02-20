import datetime
import numpy as np
import os
from typing import Literal
from tqdm import tqdm
from imageio.v2 import imread


def get_ids(ids_path: str, extension: str="") -> dict[str, int]:
    """
    Loads the information from the ids file.
    A line has the following format:
        XXXXXX.jpg <id>
    where XXXXXX represents the image number, and <id> represents the id of the person in that image.

    Returns a dictionary that maps the image number to the id of the person in that image.
    """
    ids = {}
    with open(ids_path, "r") as file:
        for line in file:
            splited_line = line.strip().split(" ")
            img_name = splited_line[0].split(".")[0]
            if extension.startswith("."):
                img_name += extension
            elif extension != "":
                img_name += "." + extension
            id = int(splited_line[1])
            ids[img_name] = id
    
    return ids



def get_num_unique_ids(ids_path: str) -> int:
    """
    Returns the number of unique ids in the ids file.
    """
    ids_map = get_ids(ids_path)
    return len(set(ids_map.values()))



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
