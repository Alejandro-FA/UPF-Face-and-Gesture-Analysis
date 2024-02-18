from Datasets.original import OriginalDataset



if __name__ == "__main__":
    original_dataset = OriginalDataset(cropped_imgs_path="data/ids_img_cropped", target_dataset_path="data/EXPANDED", annotations_path="data/")
    ids_count = original_dataset.from_cropped_to_dataset()