import Datasets as ds
import FaceRecognitionPipeline as frp
import Datasets as ds

if __name__ == '__main__':
    BASE_PATH = "data/datasets/CelebA"
    INPUT_DIR = BASE_PATH + "/Img/img_align_celeba"
    OUTPUT_DIR = BASE_PATH + "/Img/img_align_celeba_cropped"

    # Crop the images around the faces
    cropper = ds.FaceCropper(
        frp.MTCNNDetector(use_gpu=True, thresholds=(0.6, 0.7, 0.7)),
        # frp.MediaPipeDetector(model_asset_path="model/detector.tflite"),
        frp.FaceDetectorPreprocessor(output_channels=3),
        frp.FeatureExtractorPreprocessor(new_size=128, output_channels=3),
        max_faces_per_image=1,
        log_warnings=True,
        batch_size=1024,
    )
    cropper.crop(INPUT_DIR, OUTPUT_DIR, output_format="jpg") # Pass "pt" to save the images as pytorch tensors

    # Separate the images into train and test splits
    ANNOTATIONS_PATH = BASE_PATH + "/Anno/identity_CelebA.txt"
    img2id_map = ds.celeb_a.get_ids(ANNOTATIONS_PATH)
    ds.train_test_split(img2id_map, input_dir=OUTPUT_DIR, imgs_per_id_in_test=1)

    # Relabel the ids of the images
    MODIFIED_ANNOTATIONS = BASE_PATH + "/Anno/identity_CelebA_relabeled.txt"
    TRAIN_PATH = OUTPUT_DIR + "/train"
    TEST_PATH = OUTPUT_DIR + "/test"
    ds.celeb_a.relabel_ids(ANNOTATIONS_PATH, MODIFIED_ANNOTATIONS, TRAIN_PATH, TEST_PATH, percentage_no_id=0.15)
