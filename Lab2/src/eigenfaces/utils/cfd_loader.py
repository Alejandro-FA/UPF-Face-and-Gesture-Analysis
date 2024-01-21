import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from .image import Image, ImagePreprocessor
from .landmarks import Landmarks, LandmarksPreprocessor


class CFDLoader:
    """
    A class for loading images and landmarks from the CFD dataset.

    Attributes:
        LANDMARKS_CSV (str): The name of the landmarks CSV file.
        IMAGES_BASE_PATH (str): The base path for the images.
        NEUTRAL_FACE_REGEX (re.Pattern): The regular expression pattern for matching neutral face images.
    """
    LANDMARKS_CSV = "Template Database CSV 012922.csv"
    IMAGES_BASE_PATH = os.path.join('Images', 'CFD')
    NEUTRAL_FACE_REGEX = re.compile(pattern=r'[\w\-]+-N.jpg')


    def __init__(self, cfd_dataset_dir: str, landmarks_dir: str, image_preprocessor=ImagePreprocessor(), landmarks_preprocessor=LandmarksPreprocessor()) -> None:
        """
        Initializes a new instance of the CFDLoader class.

        Args:
            cfd_dataset_dir (str): The directory path of the CFD dataset.
            landmarks_dir (str): The directory path of the landmarks.
            image_preprocessor (ImagePreprocessor, optional): The image preprocessor object. Defaults to ImagePreprocessor().
            landmarks_preprocessor (LandmarksPreprocessor, optional): The landmarks preprocessor object. Defaults to LandmarksPreprocessor().
        """
        self.__images = []
        self.__landmarks = []
        self.__image_preprocessor = image_preprocessor
        self.__images_path = cfd_dataset_dir
        self.__landmarks_path = landmarks_dir
        self.__landmarks_preprocessor = landmarks_preprocessor


    def get_images(self) -> list[Image]:
        """
        Returns a list of Image objects representing the loaded images.

        Returns:
            list[Image]: A list of Image objects.
        """
        if not self.__images:
            self.__images = self.__load_images()
        return self.__images


    def get_landmarks(self) -> list[Landmarks]:
        """
        Returns a list of Landmarks objects representing the loaded landmarks.

        Returns:
            list[Landmarks]: A list of Landmarks objects.
        """
        if not self.__landmarks:
            path = os.path.join(self.__landmarks_path, self.LANDMARKS_CSV)
            self.__landmarks = self.__parse_landmarks(path)
        return self.__landmarks


    def __load_images(self) -> list[Image]:
        """
        Loads the images from the CFD dataset.

        Returns:
            list[Image]: A list of Image objects representing the loaded images.
        """
        dir = os.path.join(self.__images_path, self.IMAGES_BASE_PATH)
        images = []
        model_ids = self.__listdir_with_regex(dir, r'\w{2}-\d{3}')

        for id in tqdm(model_ids, desc='Loading images'):
            model_path = os.path.join(dir, id)
            image_names = self.__listdir_with_regex(model_path, self.NEUTRAL_FACE_REGEX)

            for image_name in image_names:
                image = Image.from_file(os.path.join(model_path, image_name), self.__image_preprocessor)
                images.append(image)

        return images
    

    def __listdir_with_regex(self, path: str, regex: str, only_dirs=False) -> list[str]:
        """
        Returns a list of directories in the given path that match the given regular expression.

        Args:
            path (str): The path in which to search for directories.
            regex (str): The regular expression to match.
            only_dirs (bool, optional): Whether to include only directories in the result. Defaults to False.

        Returns:
            list[str]: A list of directory names that match the regular expression.
        """
        return [
            dir for dir in os.listdir(path) 
            if re.match(regex, dir) and
            (os.path.isdir(os.path.join(path, dir)) or not only_dirs)
        ]
    
    
    def __parse_landmarks(self, csv_path: str) -> list[Landmarks]:
        """
        Parses the landmarks data from the specified CSV file.

        Args:
            csv_path (str): The path to the CSV file.

        Returns:
            list[Landmarks]: A list of Landmarks objects representing the parsed data.
        """
        try:
            df = pd.read_csv(csv_path)
            grouped = df.groupby('fname')
            landmarks_df = grouped.apply(lambda x: np.vstack((x['x'], x['y'])).T)
            all_paths = [f"{'/'.join(csv_path.split('/')[:-1])}/Templates/{file_path}" for file_path, _ in landmarks_df.items()]
            joint_points = self.__process_tem_files(all_paths)
            return [
                Landmarks(points, file_path, preprocessor=self.__landmarks_preprocessor, joint_points=joint_points[i])
                for i, (file_path, points) in enumerate(landmarks_df.items())
            ]
        
        except FileNotFoundError:
            print(f'Landmarks file not found: {csv_path}')
            exit(1)

    def __process_tem_files(self, file_paths: list[str]):
        joint_points = [[]] * len(file_paths)
        for idx, file_name in enumerate(file_paths):
            file_content = open(file_name).read().splitlines()
            # First 190 lines contain the landmark coordinates
            for line in file_content[190:325]:
                nums = [int(elem) for elem in line.split(" ") if elem != ""]
                if len(nums) > 0:
                    joint_points[idx].append(nums)
        
        return joint_points
                