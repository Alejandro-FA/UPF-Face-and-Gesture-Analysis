import numpy as np


class LandmarksPreprocessor:
    ORIGINAL_WIDTH = 2444
    ORIGINAL_HEIGHT = 1718
    
    def __init__(self, new_scale: tuple[int, int]=None) -> None:
        self.__new_scale = new_scale

    def preprocess(self, points: np.ndarray) -> np.ndarray:
        if self.__new_scale:
            points = LandmarksPreprocessor.__rescale(points, self.__new_scale)
            
        return points

    @staticmethod
    def __rescale(points: np.ndarray, new_scale: tuple[int, int]) -> np.ndarray:
        sf_x = new_scale[0] / LandmarksPreprocessor.ORIGINAL_WIDTH
        sf_y = new_scale[1] / LandmarksPreprocessor.ORIGINAL_HEIGHT
        new_points = np.zeros(points.shape)
        new_points[:, 0] = (points[:, 0] * sf_x)
        new_points[:, 1] = (points[:, 1] * sf_y)
        
        return new_points

class Landmarks:
    """
    A class representing facial landmarks.

    Attributes:
        __points (np.ndarray): The array of landmark points.
        __path (str): The file path associated with the landmarks.
    """


    def __init__(self, points: np.ndarray, file_path: str, preprocessor=LandmarksPreprocessor()) -> None:
        """
        Initializes the Landmarks object.

        Args:
            points (np.ndarray): The array of landmark points.
            file_path (str): The file path associated with the landmarks.
        """
        unwanted = list(range(145, 158)) + [183, 184] + list(range(135, 144))
        points = Landmarks.__remove_unwanted_points(points, unwanted)
        self.__points = preprocessor.preprocess(points).astype(int)
        self.__path = file_path


    def as_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: The landmark points as a 2D matrix (NUM_POINTS x 2).
        """
        return self.__points
    

    @property
    def path(self) -> str:
        """
        Returns:
            str: The file path associated with the landmarks.
        """
        return self.__path
    

    @staticmethod
    def __remove_unwanted_points(points, indices: list[int]) -> None:
        """
        Removes unwanted points from the landmarks.

        Args:
            indices (set[int]): A set of indices to remove from the landmarks.
        """
        points = np.delete(points, indices, axis=0)
        return points
