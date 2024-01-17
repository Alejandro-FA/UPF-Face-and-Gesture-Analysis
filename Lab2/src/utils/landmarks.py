import numpy as np

import numpy as np

class Landmarks:
    """
    A class representing facial landmarks.

    Attributes:
        NUM_POINTS (int): The number of landmarks of each face.
        __points (np.ndarray): The array of landmark points.
        __path (str): The file path associated with the landmarks.
    """
    NUM_POINTS = 189


    def __init__(self, points: np.ndarray, file_path: str) -> None:
        """
        Initializes the Landmarks object.

        Args:
            points (np.ndarray): The array of landmark points.
            file_path (str): The file path associated with the landmarks.

        Raises:
            AssertionError: If the shape of the points array is not (NUM_POINTS, 2).
        """
        assert(points.shape == (self.NUM_POINTS, 2))
        self.__points = points
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
