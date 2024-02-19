import os
from .utils import get_log_path
from tqdm import tqdm


def __log_stats(imgs_per_id: dict[int, int], imgs_per_id_in_test: int):
    """
    Logs the statistics of the dataset split.

    Args:
        imgs_per_id: dictionary that maps image ids to the number of images per id
        imgs_per_id_in_test: number of images per id included in the test set
    """
    log_path = get_log_path("split", extension="log")

    with open(log_path, "w") as file:
        unique_test_ids = set()
        unique_train_ids = set()
        for id, count in imgs_per_id.items():
            if count == imgs_per_id_in_test:
                unique_test_ids.add(id)
            elif count > imgs_per_id_in_test:
                unique_test_ids.add(id)
                unique_train_ids.add(id)

        file.write(f"Unique test ids: {len(unique_test_ids)}\n")
        file.write(f"Unique train ids: {len(unique_train_ids)}\n")

        for id, count in imgs_per_id.items():
            file.write(f"ID: {id}, test count: {imgs_per_id_in_test}, train count: {count - imgs_per_id_in_test}\n")


def __get_images_paths(directory: str, extension: str) -> list[str]:
    """
    Returns the paths of the images in the given directory.

    Args:
        directory: directory containing the images
        extension: file extension of the images
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Invalid directory {directory}")

    images_paths = []
    for img in os.listdir(directory):
        path = os.path.join(directory, img)
        if os.path.isfile(path) and img.endswith(extension):
            images_paths.append(img)
    return images_paths



def train_test_split(img2id_map: dict[str, int], input_dir, imgs_per_id_in_test: int = 1, log_results=True) -> None:
    """
    Splits the dataset into a train and test set. Creates a new directory for
    each set, and moves the images accordingly.

    Args:
        img2id_map: dictionary that maps image filenames to ids
        input_dir: directory containing the images
        imgs_per_id_in_test: number of images per id to include in the test set. All other images are included in the train set.
        log_results: flag indicating whether to log the results of the dataset split

    Returns:
        None
    """
    # Create the train and test directories
    train_dir = os.path.join(input_dir, "train")
    test_dir = os.path.join(input_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create a dictionary to keep track of the number of images per id in the test set
    imgs_per_id: dict[int, int] = {}
    for id in img2id_map.values():
        imgs_per_id[id] = 0

    # Move the images to the train and test directories
    images_paths = __get_images_paths(input_dir, "jpg")

    for img in tqdm(images_paths, desc="Splitting dataset into train and test sets"):
        img_name = img.split(".")[0]
        id = img2id_map[img_name]

        if imgs_per_id[id] >= imgs_per_id_in_test:
            os.rename(os.path.join(input_dir, img), os.path.join(train_dir, img))
        else:
            os.rename(os.path.join(input_dir, img), os.path.join(test_dir, img))

        imgs_per_id[id] += 1

    # Move the ids that are not present in both the train and test sets to "unused" directory
    test_images_paths = __get_images_paths(test_dir, "jpg")
    unused_dir = os.path.join(input_dir, "unused")
    if len(test_images_paths) > 0:
        os.makedirs(unused_dir, exist_ok=True)

    for img in tqdm(test_images_paths, desc="Moving ids not present in both splits to 'unused' directory"):
        img_name = img.split(".")[0]
        id = img2id_map[img_name]
        if imgs_per_id[id] == 0 or imgs_per_id[id] == imgs_per_id_in_test:
            os.rename(os.path.join(test_dir, img), os.path.join(unused_dir, img))
        
    # Log the results
    if log_results and len(images_paths) > 0:
        __log_stats(imgs_per_id, imgs_per_id_in_test)
