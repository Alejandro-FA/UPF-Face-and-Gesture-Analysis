import numpy as np


class LandmarksPreprocessor:
    """
    A class for preprocessing facial landmarks.

    Attributes:
        ORIGINAL_WIDTH (int): The original width of the image.
        ORIGINAL_HEIGHT (int): The original height of the image.

    Args:
        new_scale (tuple[int, int], optional): The new scale to rescale the landmarks to. Defaults to None.
        unwanted_points (list[int], optional): The list of unwanted points to remove from the landmarks. Defaults to [].
    """
    ORIGINAL_WIDTH = 2444 # FIXME: This should be a parameter
    ORIGINAL_HEIGHT = 1718

    def __init__(self, new_scale: tuple[int, int] = None, unwanted_points: list[int] = []) -> None:
        self.__new_scale = new_scale
        self.__unwanted_points = unwanted_points

    def preprocess(self, points: np.ndarray) -> np.ndarray:
        """
        Preprocesses the facial landmarks.

        Args:
            points (np.ndarray): The array of landmark points.

        Returns:
            np.ndarray: The preprocessed landmark points.
        """
        if self.__new_scale:
            points = self.__rescale(points, self.__new_scale)
        if self.__unwanted_points:
            points = self.__remove_unwanted(points, self.__unwanted_points)
        return points

    @staticmethod
    def __rescale(points: np.ndarray, new_scale: tuple[int, int]) -> np.ndarray:
        """
        Rescales the facial landmarks.

        Args:
            points (np.ndarray): The array of landmark points.
            new_scale (tuple[int, int]): The new scale to rescale the landmarks to.

        Returns:
            np.ndarray: The rescaled landmark points.
        """
        sf_x = new_scale[0] / LandmarksPreprocessor.ORIGINAL_WIDTH
        sf_y = new_scale[1] / LandmarksPreprocessor.ORIGINAL_HEIGHT
        new_points = np.zeros(points.shape)
        new_points[:, 0] = (points[:, 0] * sf_x)
        new_points[:, 1] = (points[:, 1] * sf_y)

        return new_points

    @staticmethod
    def __remove_unwanted(points: np.ndarray, unwanted_points: list[int]) -> np.ndarray:
        """
        Removes unwanted points from the facial landmarks.

        Args:
            points (np.ndarray): The array of landmark points.
            unwanted_points (list[int]): The list of unwanted points to remove.

        Returns:
            np.ndarray: The landmark points with unwanted points removed.
        """
        return np.delete(points, unwanted_points, axis=0)


class Landmarks:
    """
    A class representing facial landmarks.

    Attributes:
        __points (np.ndarray): The array of landmark points.
        __path (str): The file path associated with the landmarks.

    Args:
        points (np.ndarray): The array of landmark points.
        file_path (str): The file path associated with the landmarks.
        preprocessor (LandmarksPreprocessor, optional): The preprocessor for the landmarks. Defaults to LandmarksPreprocessor().
    """

    def __init__(self, points: np.ndarray, file_path: str, preprocessor=LandmarksPreprocessor()) -> None:
        """
        Initializes the Landmarks object.

        Args:
            points (np.ndarray): The array of landmark points.
            file_path (str): The file path associated with the landmarks.
            preprocessor (LandmarksPreprocessor, optional): The preprocessor for the landmarks. Defaults to LandmarksPreprocessor().
        """
        self.__points = preprocessor.preprocess(points).astype(int)
        self.__path = file_path

    def as_vector(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: The landmark points as a 1D vector.
        """
        return self.__points.flatten()
    
    
    def as_matrix(self) -> np.ndarray:
        """
        Returns the landmark points as a 2D matrix.

        Returns:
            np.ndarray: The landmark points as a 2D matrix (NUM_POINTS x 2).
        """
        return self.__points


    @property
    def path(self) -> str:
        """
        Returns the file path associated with the landmarks.

        Returns:
            str: The file path associated with the landmarks.
        """
        return self.__path
