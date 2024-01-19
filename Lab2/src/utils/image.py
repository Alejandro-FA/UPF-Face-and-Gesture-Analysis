import numpy as np
import cv2



class ImagePreprocessor:
    def __init__(self, new_size: tuple[int, int]=None, new_color=None) -> None:
        self.__new_size = new_size
        self.__new_color = new_color


    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if self.__new_size:
            image = ImagePreprocessor.__resize(image, self.__new_size)
        if self.__new_color:
            image = ImagePreprocessor.__convert_color(image, self.__new_color)
        return image


    @staticmethod
    def __resize(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
        return cv2.resize(image, new_size)


    @staticmethod
    def __convert_color(image: np.ndarray, new_color: int) -> np.ndarray:
        return cv2.cvtColor(image, new_color)
    




class Image:
    """
    A class representing an image.

    Attributes:
        __pixels (np.ndarray): The pixel values of the image.
        __path (str): The path of the image file.
    """
    
    def __init__(self, data: np.ndarray, input_path: str=None, preprocessor=ImagePreprocessor()) -> None:
        self.__pixels: np.ndarray = preprocessor.preprocess(data)
        self.__path: str = input_path


    def from_file(input_path: str, preprocessor=ImagePreprocessor()) -> 'Image':
        data = cv2.imread(input_path)
        if data is None:
            print(f'Failed to read image from path: {input_path}')
            exit(1)
        return Image(data, input_path, preprocessor)


    @staticmethod
    def from_vector(vector: np.ndarray, image_size: int, input_path: str=None, preprocessor=ImagePreprocessor()) -> 'Image':
        assert vector.ndim == 1, 'The input data is not a 1D vector.'
        width, height = image_size
        data = vector.reshape(height, width)
        return Image(data, input_path, preprocessor)


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
    

    def copy(self) -> 'Image':
        """
        Returns:
            Image: A copy of the image.
        """
        return Image(self.__pixels.copy(), self.__path)


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

    
    def show(self, title: str=None) -> None:
        """
        Displays the image.
        """
        title = title if title else self.__path
        cv2.imshow(title, self.__pixels)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
