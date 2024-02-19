import os
from .utils import get_ids


def get_unique_ids_from_dir(images_dir: str, ids: dict[str, int]) -> set[int]:
    unique_ids = set()
    for image in sorted(os.listdir(images_dir)):
        image_num = image.split(".")[0]
        unique_ids.add(ids[image_num])
    return unique_ids



def relabel_ids(original_annotations_path: str, output_annotations_path: str, train_images_dir: str, test_images_dir:str):
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

    if len(mismatched_ids) != 0:
        print("The following ids are not present in both splits:", mismatched_ids)
        raise ValueError("The ids in the training and testing dataset must be the same")

        
    # Remap the ids to be consecutive numbers starting from 0, with no id being 0
    # Needed to be compatible with the PyTorch CrossEntropyLoss
    old2new_ids = {id: i for i, id in enumerate(sorted(set(ids.values())))}
    new_unique_ids = sorted(old2new_ids.values())
    
    # Write the new annotations file
    print(f"Writing the new annotations file to {output_annotations_path}...") # TODO: Move this section to a separate function
    with open(output_annotations_path, "w") as output_file:
        for img_name, old_id in ids.items():
            new_id = old2new_ids[old_id]
            output_file.write(f"{img_name}.jpg {new_id}\n")

    # Print some statistics
    print("Current number of unique ids:", len(new_unique_ids))
    print("Max id:", max(new_unique_ids), ". Min ids:", min(new_unique_ids))