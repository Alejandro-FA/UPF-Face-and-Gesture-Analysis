import numpy as np
import random

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


if __name__ == "__main__":
    ANNOTATIONS_PATH = "data/datasets/CelebA/Anno/identity_CelebA.txt"
    MODIFIED_ANNOTATIONS = "data/datasets/CelebA/Anno/identity_CelebA_relabeled.txt"
    
    ids = get_ids(ANNOTATIONS_PATH)
    unique_ids = list(np.unique(list(ids.values())))
    
    percentage_no_id = 0.15
    removed_ids = random.sample(unique_ids, round(len(unique_ids) * percentage_no_id))
    
    modified_images = 0
    with open(MODIFIED_ANNOTATIONS, "w") as output_file:
        for img_name, id in ids.items():
            if id in removed_ids:
                output_file.write(f"{img_name}.jpg 0\n")
                modified_images += 1
            else:
                output_file.write(f"{img_name}.jpg {id}\n")
    
    print(f"The label of [{modified_images}/{len(ids)}] ({round(modified_images/len(ids) * 100, 2)} %) images have been set to 0")