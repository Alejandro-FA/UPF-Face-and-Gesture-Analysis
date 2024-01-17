import numpy as np
import cv2

class Image:
    """
    A class representing an image.

    Attributes:
        __pixels (np.ndarray): The pixel values of the image.
        __path (str): The path of the image file.
    """
    
    def __init__(self, input_path: str) -> None:
        """
        Initializes the Image object.

        Args:
            input_path (str): The path of the image file.
        """
        self.__pixels: np.ndarray = cv2.imread(input_path)
        if self.__pixels is None:
            print(f'Failed to read image from path: {input_path}')
            exit(1)
        self.__path = input_path


    def as_vector(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: The pixels of the image as a 1D-vector.
        """
        return self.__pixels.flatten()
    

    def as_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: The pixels of the image as a 3D-matrix (width x height x channels).
        """
        return self.__pixels


    @property
    def width(self) -> int:
        """
        Returns:
            int: The width of the image.
        """
        return self.__pixels.shape[1]


    @property
    def height(self) -> int:
        """
        Returns:
            int: The height of the image.
        """
        return self.__pixels.shape[0]


    @property
    def path(self) -> str:
        """
        Returns:
            str: The path of the image file.
        """
        return self.__path

    
    def show(self) -> None:
        """
        Displays the image.
        """
        cv2.imshow(self.__path, self.__pixels)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
