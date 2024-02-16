import os

def get_ids(ids_path: str) -> dict[str, int]:
    """
    Loads the information from the ids file.
    A line has the following format:
        XXXXXX.jpg <id>
    where XXXXXX represents the image number, and <id> represents the id of the person in that image.
    """
    ids = {}
    try:
        file = open(ids_path, "r").read().strip()
    except:
        raise FileNotFoundError(f"Filt {ids_path} could not be found")

    for line in file.splitlines():
        splited_line = line.split(" ")
        img_num = splited_line[0].split(".")[0]
        id = int(splited_line[1])
        ids[img_num] = id
    
    return ids


def label_split(dir_path: str, ids: dict[str, int], labels_output_path: str):
    """"
    Creates the annotations file for the images given from dir_path.
    """
    file_list = sorted(os.listdir(dir_path))
    with open(labels_output_path, "w") as file:
        for image_path in file_list:
            image_name = image_path.split(".")[0]
            file.write(f"{image_name}.jpg {ids[image_name]}\n")
        


if __name__ == "__main__":
    TRAINING_SET_PATH = "data/datasets/CelebA/Img/img_align_celeba_train"
    TESTING_SET_PATH = "data/datasets/CelebA/Img/img_align_celeba_test"
    ANNOTATIONS_PATH = "data/datasets/CelebA/Anno/identity_CelebA_relabeled.txt"
    
    TRAIN_ANNOTATIONS_PATH = "data/datasets/CelebA/Anno/identity_CelebA_train.txt"
    TEST_ANNOTATIONS_PATH = "data/datasets/CelebA/Anno/identity_CelebA_test.txt"
    
    ids = get_ids(ANNOTATIONS_PATH)
    
    label_split(TRAINING_SET_PATH, ids, TRAIN_ANNOTATIONS_PATH)
    label_split(TESTING_SET_PATH, ids, TEST_ANNOTATIONS_PATH)