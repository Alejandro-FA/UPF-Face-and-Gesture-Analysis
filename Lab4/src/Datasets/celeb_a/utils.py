import os
import random


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



def get_unique_ids_from_dir(images_dir: str, ids: dict[str, int]) -> set[int]:
    unique_ids = set()
    for image in sorted(os.listdir(images_dir)):
        image_num = image.split(".")[0]
        unique_ids.add(ids[image_num])
    return unique_ids



def relabel_ids(original_annotations_path: str, output_annotations_path: str, train_images_dir: str, test_images_dir:str, percentage_no_id: float = 0.15):
    print("Relabeling ids...")

    # Get the ids from the original annotations file 
    ids = get_ids(original_annotations_path)
    
    # Find the ids that are not present in both the training and testing dataset
    train_ids = get_unique_ids_from_dir(train_images_dir, ids)
    test_ids = get_unique_ids_from_dir(test_images_dir, ids)
    matched_ids = test_ids.intersection(train_ids)
    mismatched_ids = test_ids.difference(train_ids)
    print("Number of ids present in both splits:", len(matched_ids))
    print("Number of ids not present in both splits:", len(mismatched_ids))

    # Set all the mismatched ids and a percentage of the unique ids to -1
    removed_ids = random.sample(sorted(matched_ids), round(len(matched_ids) * percentage_no_id))
    removed_ids.extend(mismatched_ids)

    modified_images = 0
    for img_name, id in ids.items():
        if id in removed_ids:
            ids[img_name] = -1 # Use a negative number to avoid conflicts with the old ids
            modified_images += 1
        
    # Remap the ids to be consecutive numbers starting from 0, with no id being 0
    # Needed to be compatible with the PyTorch CrossEntropyLoss
    old2new_ids = {id: i for i, id in enumerate(sorted(set(ids.values())))}
    new_unique_ids = sorted(old2new_ids.values())
    
    # Write the new annotations file
    print(f"Writing the new annotations file to {output_annotations_path}...")
    with open(output_annotations_path, "w") as output_file:
        for img_name, old_id in ids.items():
            new_id = old2new_ids[old_id]
            output_file.write(f"{img_name}.jpg {new_id}\n")

    # Print some statistics
    print(f"The label of [{modified_images}/{len(ids)}] ({round(modified_images/len(ids) * 100, 2)} %) images have been set to 0")
    print("Current number of unique ids:", len(new_unique_ids))
    print("Max id:", max(new_unique_ids), ". Min ids:", min(new_unique_ids))
    