import numpy as np
import cv2


class ImagePreprocessor:
    def __init__(self, new_size: tuple[int, int] = None, new_color=None) -> None:
        """
        Initialize the ImagePreprocessor class.

        Args:
            new_size (tuple[int, int], optional): The new size of the image. Defaults to None.
            new_color (optional): The new color space of the image. Defaults to None.
        """
        self.__new_size = new_size
        self.__new_color = new_color

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The preprocessed image.
        """
        if self.__new_size:
            image = ImagePreprocessor.__resize(image, self.__new_size)
        if self.__new_color:
            image = ImagePreprocessor.__convert_color(image, self.__new_color)
        return image

    @staticmethod
    def __resize(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
        """
        Resizes the image.

        Args:
            image (np.ndarray): The input image.
            new_size (tuple[int, int]): The new size of the image.

        Returns:
            np.ndarray: The resized image.
        """
        return cv2.resize(image, new_size)

    @staticmethod
    def __convert_color(image: np.ndarray, new_color: int) -> np.ndarray:
        """
        Converts the color space of the image.

        Args:
            image (np.ndarray): The input image.
            new_color: The new color space of the image.

        Returns:
            np.ndarray: The image with the new color space.
        """
        return cv2.cvtColor(image, new_color)


class Image:
    """
    A class representing an image.

    Attributes:
        __pixels (np.ndarray): The pixel values of the image.
        __path (str): The path of the image file.
    """
    def __init__(self, data: np.ndarray, input_path: str = None, preprocessor=ImagePreprocessor()) -> None:
        """
        Initialize the Image class.

        Args:
            data (np.ndarray): The pixel values of the image.
            input_path (str, optional): The path of the image file. Defaults to None.
            preprocessor (ImagePreprocessor, optional): The image preprocessor. Defaults to ImagePreprocessor().
        """
        data = Image.__clamp(data, min_val=0, max_val=255)
        data = data.astype(np.uint8)
        self.__pixels: np.ndarray = preprocessor.preprocess(data)
        self.__path: str = input_path

    @staticmethod
    def from_file(input_path: str, preprocessor=ImagePreprocessor()) -> 'Image':
        """
        Create an Image object from an image file.

        Args:
            input_path (str): The path of the image file.
            preprocessor (ImagePreprocessor, optional): The image preprocessor. Defaults to ImagePreprocessor().

        Returns:
            Image: The Image object.
        """
        data = cv2.imread(input_path)
        if data is None:
            print(f'Failed to read image from path: {input_path}')
            exit(1)
        return Image(data, input_path, preprocessor)

    @staticmethod
    def from_vector(vector: np.ndarray, image_size: int, input_path: str = None, preprocessor=ImagePreprocessor()) -> 'Image':
        """
        Create an Image object from a vector.

        Args:
            vector (np.ndarray): The pixel values of the image as a 1D vector.
            image_size (int): The size of the image.
            input_path (str, optional): The path of the image file. Defaults to None.
            preprocessor (ImagePreprocessor, optional): The image preprocessor. Defaults to ImagePreprocessor().

        Returns:
            Image: The Image object.
        """
        assert vector.ndim == 1, 'The input data is not a 1D vector.'
        width, height = image_size
        data = vector.reshape(height, width)
        return Image(data, input_path, preprocessor)

    def as_vector(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: The pixels of the image as a 1D vector.
        """
        return self.__pixels.flatten()

    def as_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: The pixels of the image as a 3D matrix (width x height x channels).
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

    def show(self, title: str = None) -> None:
        """
        Displays the image.

        Args:
            title (str, optional): The title of the image window. If no title is passed, the path of the image file will be used. Defaults to None.
        """
        title = title if title else self.__path if self.__path else 'Image'
        cv2.imshow(title, self.__pixels)
        while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) > -1:
                cv2.destroyAllWindows()
                break

    @staticmethod
    def __clamp(data: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
        """
        Clamps the pixel values of the image.

        Args:
            data (np.ndarray): The pixel values of the image.
            min_val (int): The minimum pixel value.
            max_val (int): The maximum pixel value.

        Returns:
            np.ndarray: The clamped image.
        """
        return np.clip(data, min_val, max_val)
