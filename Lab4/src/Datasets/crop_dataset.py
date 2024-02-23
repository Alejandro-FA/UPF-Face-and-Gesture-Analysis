from typing import Literal
import FaceRecognitionPipeline as frp
from imageio.v2 import imread, imwrite
import torch
import os
from tqdm import tqdm
import imageio.v2
from .utils import get_log_path


class FaceCropper:
    """
    A class that crops faces from images using a face detection pipeline.

    Args:
        face_detector (frp.FaceDetector): The face detector model.
        preprocessor (frp.FaceDetectorPreprocessor): The preprocessor for face detection.
        postprocessor (frp.FeatureExtractorPreprocessor): The postprocessor for face detection.
        max_faces_per_image (int, optional): The maximum number of faces to detect per image. Defaults to 1.
        log_warnings (bool, optional): Whether to log warnings. Defaults to True.
        batch_size (int, optional): The batch size for processing images. Defaults to 128.
    """

    def __init__(self, face_detector: frp.FaceDetector, preprocessor: frp.FaceDetectorPreprocessor, postprocessor: frp.FeatureExtractorPreprocessor, max_faces_per_image: int = 1, log_warnings=True, batch_size=128, only_one_detection: bool = False):
        self.face_detector = face_detector
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.max_faces_per_image = max_faces_per_image
        self.log_warnings = log_warnings
        self.log_base_path = "crop"
        self.batch_size = batch_size
        self.only_one_detection = only_one_detection


    def crop(self, input_dir: str, output_dir: str, output_format: Literal["jpg", "png", "pt"]):
        """
        Crop faces from images in the input directory and save them in the output directory.

        Args:
            input_dir (str): The directory containing the input images.
            output_dir (str): The directory to save the cropped face images.
            output_format (Literal["jpg", "png", "pt"]): The output image format.

        Returns:
            None
        """
        log_path = get_log_path(self.log_base_path, extension="log")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        batch = []
        image_paths = []
        for image_path in os.listdir(input_dir):
            if os.path.isdir(f"{input_dir}/{image_path}"): continue # Avoid processing any directory in the input directory
            try:
                image = imread(f"{input_dir}/{image_path}")
                image = self.preprocessor(image)
                batch.append(image)
                image_paths.append(image_path)
            except:
                print(f"Omitting {image_path} as it cannot be opened")

            if len(batch) == self.batch_size:
                self.__process_batch(batch, image_paths, output_dir, output_format, log_path)
                batch = []
                image_paths = []
        
        if len(batch) > 0:
            self.__process_batch(batch, image_paths, output_dir, output_format, log_path)


    def __process_batch(self, batch: list[imageio.v2.Array], image_paths: list[str], output_dir: str, output_format: Literal["jpg", "png", "pt"], log_path: str):
        """
        Process a batch of images and save the cropped faces.

        Args:
            batch (list[imageio.v2.Array]): The batch of images.
            image_paths (list[str]): The paths of the images.
            output_dir (str): The directory to save the cropped face images.
            output_format (Literal["jpg", "png", "pt"]): The output image format.
            log_path (str): The path to the log file.

        Returns:
            None
        """
        try:
            results = self.face_detector(batch)
        except:
            raise Exception
            # FIXME: What is this for? Ideally face_detector should not raise an exception. Solve the source of the exception.
            return
        if results == []: # Batch with no detections
            return
        if not isinstance(results[0], list):
            results = [results]

        for i, image in enumerate(batch):
            results_i = results[i]
            if len(results_i) > 1 and self.only_one_detection:
                continue
            results_i = self.__reduce_results(results_i, image_paths[i], log_path)
            for j, res in enumerate(results_i):
                face_image = self.postprocessor(image, res.bounding_box)
                suffix = f"_{j}" if j > 0 else ""
                output_base_path = f"{output_dir}/{image_paths[i].split('.')[0]}{suffix}"
                self.__save_image(face_image, output_base_path, output_format)


    def __save_image(self, image: imageio.v2.Array, output_base_path: str, output_format: Literal["jpg", "png", "pt"]):
        """
        Save the cropped face image.

        Args:
            image (imageio.v2.Array): The cropped face image.
            output_base_path (str): The base path for the output image.
            output_format (Literal["jpg", "png", "pt"]): The output image format.

        Returns:
            None
        """
        if output_format == "pt":
            tensor = torch.from_numpy(image)
            torch.save(tensor, f"{output_base_path}.pt")
        else:
            imwrite(f"{output_base_path}.{output_format}", image)


    def __reduce_results(self, results: list[frp.DetectionResult], image_path: str, log_path: str) -> list[frp.DetectionResult]:
        """
        Reduces the number of faces in the results to self.max_faces_per_image.

        Args:
            results (list[frp.DetectionResult]): The list of face detection results.
            image_path (str): The path of the image.
            log_path (str): The path to the log file.

        Returns:
            list[frp.DetectionResult]: The reduced list of face detection results.
        """
        n = len(results)
        if self.log_warnings:
            with open(log_path, "a") as file:
                if n > self.max_faces_per_image:
                    file.write(f"Warning: Image {image_path} has more than {self.max_faces_per_image} faces. Only the first {self.max_faces_per_image} faces have been saved.\n")
                elif n == 0:
                    file.write(f"Warning: Image {image_path} has no faces.\n")

        return results[:self.max_faces_per_image]
    