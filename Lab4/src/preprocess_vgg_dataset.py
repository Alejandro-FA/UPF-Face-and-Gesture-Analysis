"""
This script has to be used after downloading the images of the famous people with the web scraper.
It generates another directory with the same structure and images, but cropped.
"""

import Datasets as ds
import FaceRecognitionPipeline as frp
import Datasets as ds
import os
from Datasets.original import OriginalDatasetSplitter
import argparse
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preprocess images from the original dataset.')
    parser.add_argument('--crop', '-c', action='store_true', default=False, help='Crop the images')
    parser.add_argument('--expand', '-e', action='store_true', default=False, help='Expand the images')
    parser.add_argument('--split', '-s', action='store_true', default=False, help='Split the images into train-test splits')
    parser.add_argument('--relabel', '-r', action='store_true', default=False, help='Relabel the ids of the images')
    args = parser.parse_args()

    if not args.crop and not args.expand and not args.split and not args.relabel:
        args.expand = True
        args.split = True
        args.relabel = True
    return args



if __name__ == "__main__":
    args = parse_arguments()
    BASE_PATH = "data/datasets/VGG-Face2"
    ANNOTATIONS_PATH = BASE_PATH + "/annotations.txt"
    OUTPUT_DIR = BASE_PATH + "/cropped"

    if args.crop:
        INPUT_DIR = BASE_PATH + "/data/images"
        cropper = ds.FaceCropper(
            frp.MTCNNDetector(use_gpu=False, thresholds=(0.6, 0.7, 0.7)),
            # frp.MediaPipeDetector(model_asset_path="model/detector.tflite"),
            frp.FaceDetectorPreprocessor(output_channels=3), 
            frp.FeatureExtractorPreprocessor(new_size=128, output_channels=3),
            max_faces_per_image=4,
            log_warnings=False,
            batch_size=1,
            only_one_detection=True
        )
        
        all_dirs = os.listdir(INPUT_DIR)
        for i, dir in tqdm(enumerate(all_dirs), desc="Cropping images", total=len(all_dirs)):
            if dir == ".DS_Store": continue
            dir_path = INPUT_DIR + f"/{dir}"
            # print(f"[{i + 1} / {len(all_dirs)}]. Processing images of {dir}...")
            cropper.crop(dir_path, f"{INPUT_DIR}_cropped/{dir}", output_format="jpg") # Pass "pt" to save the images as pytorch tensors

    # Manual step required here to remove the faces that do not belong to the
    # appropriate person.

    if args.expand:
        original_dataset = OriginalDatasetSplitter(cropped_imgs_path="data/ids_img_cropped", target_dataset_path=OUTPUT_DIR, annotations_path=ANNOTATIONS_PATH)
        ids_count = original_dataset.from_cropped_to_dataset()

    if args.split: # Separate the images into train and test splits
        img2id_map = ds.get_ids(ANNOTATIONS_PATH)
        ds.train_test_split(img2id_map, input_dir=OUTPUT_DIR, imgs_per_id_in_test=2)

    if args.relabel:
        # Relabel the ids of the images
        MODIFIED_ANNOTATIONS = "data/expanded_annotations_relabeled.txt"
        TRAIN_IMAGES_DIR = OUTPUT_DIR + "/train"
        TEST_IMAGES_DIR = OUTPUT_DIR + "/test"
        ds.relabel_ids(ANNOTATIONS_PATH, MODIFIED_ANNOTATIONS, TRAIN_IMAGES_DIR, TEST_IMAGES_DIR)
    