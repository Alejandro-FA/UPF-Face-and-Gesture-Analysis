import numpy as np
import random
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


def get_train_ids(training_images_dir: str, ids: dict[str, int]) -> list[int]:
    train_ids = []
    for image in sorted(os.listdir(training_images_dir)):
        image_num = image.split(".")[0]
        train_ids.append(ids[image_num])

    return list(set(train_ids))



if __name__ == "__main__":
    ANNOTATIONS_PATH = "data/datasets/CelebA/Anno/identity_CelebA.txt"
    MODIFIED_ANNOTATIONS = "data/datasets/CelebA/Anno/identity_CelebA_relabeled.txt"
    TRAIN_PATH = "data/datasets/CelebA/Img/img_align_celeba_train"
    TEST_PATH = "data/datasets/CelebA/Img/img_align_celeba_test"
    
    ids = get_ids(ANNOTATIONS_PATH)
    unique_ids = set(ids.values())
    
    percentage_no_id = 0.15
    removed_ids = random.sample(unique_ids, round(len(unique_ids) * percentage_no_id))
    train_ids = get_train_ids(TRAIN_PATH, ids)
    ids_not_in_train = [id for id in unique_ids if id not in train_ids]
    print("Ids present in test dataset but not in train:", len(ids_not_in_train))
    removed_ids.extend(ids_not_in_train)

    new_unique_ids = unique_ids - set(removed_ids)
    ids_map = {id: i+1 for i, id in enumerate(new_unique_ids)}
    
    modified_images = 0
    with open(MODIFIED_ANNOTATIONS, "w") as output_file:
        for img_name, id in ids.items():
            if id in removed_ids:
                output_file.write(f"{img_name}.jpg 0\n")
                modified_images += 1
            else:
                output_file.write(f"{img_name}.jpg {ids_map[id]}\n")
    
    print(f"The label of [{modified_images}/{len(ids)}] ({round(modified_images/len(ids) * 100, 2)} %) images have been set to 0")
    new_unique_ids = set(get_ids(MODIFIED_ANNOTATIONS).values())
    
    print("Current number of unique ids:", len(new_unique_ids))
    print("Max id:", max(new_unique_ids), ". Min ids:", min(new_unique_ids))
    print("Ids sample:", sorted(new_unique_ids)[:20])