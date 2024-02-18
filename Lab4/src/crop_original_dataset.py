import Datasets as ds
import FaceRecognitionPipeline as frp
import Datasets as ds
import os


if __name__ == "__main__":
    INPUT_DIR = "data/ids_img"
    cropper = ds.FaceCropper(
        frp.MTCNNDetector(use_gpu=False, thresholds=(0.6, 0.7, 0.7)),
        # frp.MediaPipeDetector(model_asset_path="model/detector.tflite"),
        frp.FaceDetectorPreprocessor(output_channels=3),
        frp.FeatureExtractorPreprocessor(new_size=128, output_channels=3),
        max_faces_per_image=4,
        log_warnings=False,
        batch_size=1,
    )
    
    all_dirs = os.listdir(INPUT_DIR)
    for i, dir in enumerate(all_dirs):
        if dir == ".DS_Store": continue
        dir_path = INPUT_DIR + f"/{dir}"
        print(f"[{i + 1} / {len(all_dirs)}]. Processing images of {dir}...")
        cropper.crop(dir_path, f"{INPUT_DIR}_cropped/{dir}", output_format="jpg") # Pass "pt" to save the images as pytorch tensors
    