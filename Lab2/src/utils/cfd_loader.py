import os
import re
import pandas as pd
from .image import Image
from .landmarks import Landmarks
import numpy as np
from tqdm import tqdm


class LandmarksParser:
    """
    A class for parsing landmarks data from a CSV file.
    """

    def parse(self, csv_path: str) -> list[Landmarks]:
        """
        Parses the landmarks data from the specified CSV file.

        Args:
            csv_path (str): The path to the CSV file.

        Returns:
            list[Landmarks]: A list of Landmarks objects representing the parsed data.
        """
        try:
            df = pd.read_csv(csv_path)
            return self.__parse_df(df)
        except FileNotFoundError:
            print(f'Landmarks file not found: {csv_path}')
            exit(1)


    def __parse_df(self, df: pd.DataFrame) -> list[Landmarks]:
        """
        Parses the landmarks data from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the landmarks data.

        Returns:
            list[Landmarks]: A list of Landmarks objects representing the parsed data.
        """
        grouped = df.groupby('fname')
        landmarks_df = grouped.apply(lambda x: np.vstack((x['x'], x['y'])).T)
        return [Landmarks(points, file_path) for file_path, points in landmarks_df.items()]

 



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


    def __init__(self, cfd_dataset_dir: str, landmarks_dir: str, landmarks_parser=LandmarksParser()) -> None:
        """
        Initializes a new instance of the CFDLoader class.

        Args:
            cfd_dataset_dir (str): The directory path of the CFD dataset.
            landmarks_dir (str): The directory path of the landmarks.
            landmarks_parser (LandmarksParser, optional): The landmarks parser object. Defaults to LandmarksParser().
        """
        self.__images = []
        self.__landmarks = []
        self.__landmarks_parser = landmarks_parser
        self.__images_path = cfd_dataset_dir
        self.__landmarks_path = landmarks_dir


    def get_images(self) -> list[Image]:
        """
        Returns:
            list[Image]: A list of Image objects representing the loaded images.
        """
        if not self.__images:
            self.__images = self.__load_images()
        return self.__images


    def get_landmarks(self) -> list[Landmarks]:
        """
        Returns:
            list[Landmarks]: A list of Landmarks objects representing the loaded landmarks.
        """
        if not self.__landmarks:
            path = os.path.join(self.__landmarks_path, self.LANDMARKS_CSV)
            self.__landmarks = self.__landmarks_parser.parse(path)
        return self.__landmarks


    def __load_images(self) -> list[Image]:
        """
        Loads the images from the CFD dataset.

        Returns:
            list[Image]: A list of Image objects representing the loaded images.
        """
        dir = os.path.join(self.__images_path, self.IMAGES_BASE_PATH)
        images = []
        for model_id in tqdm(os.listdir(dir), desc='Loading images'):
            model_path = os.path.join(dir, model_id)
            for image_name in os.listdir(model_path):
                if re.match(self.NEUTRAL_FACE_REGEX, image_name):
                    image = Image(os.path.join(model_path, image_name))
                    images.append(image)
                    continue
        return images
    