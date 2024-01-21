import numpy as np
import cv2


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

    def preprocess_points(self, points: np.ndarray):
        """
        Preprocesses the facial landmarks.

        Args:
            points (np.ndarray): The array of landmark points.

        Returns:
            np.ndarray: The preprocessed landmark points.
        """
        index_mapping = None
        if self.__new_scale:
            points = self.__rescale(points, self.__new_scale)
        if self.__unwanted_points:
            points, index_mapping = self.__remove_unwanted(points, self.__unwanted_points)
        return points, index_mapping
    
    def preprocess_joint_points(self, joint_points: list[list[int]]) -> list[list[int]]:
        """
        Preprocesses the facial landmarks.

        Args:
            joint_points (list[list[int]]): A list of lines, which represent the points that are connected in a landmark. Each line contains the index of the landmark points that are connected

        Returns:
            list[list[int]]: The preprocessed joint points.
        """
        for i in range(len(joint_points)):
            joint_points[i] = [val for val in joint_points[i] if val not in self.__unwanted_points]
        
        return joint_points

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
    def __remove_unwanted(points: np.ndarray, unwanted_points: list[int]):
        """
        Removes unwanted points from the facial landmarks.

        Args:
            points (np.ndarray): The array of landmark points.
            unwanted_points (list[int]): The list of unwanted points to remove.

        Returns:
            np.ndarray: The landmark points with unwanted points removed.
        """
        original_points = points.copy()
        filtered_points = np.delete(points, unwanted_points, axis=0)
        
        index_mapping = {}
        
        for i in range(filtered_points.shape[0]):
            original_idx = LandmarksPreprocessor.__find_row_index(original_points, filtered_points[i, :])
            if original_idx is not None:
                index_mapping[original_idx[0]] = i
        
        return filtered_points, index_mapping

    
    @staticmethod
    def __find_row_index(array: np.ndarray, target_row: np.ndarray):
        indices = np.where(np.all(array == target_row, axis=1))
        return indices[0] if len(indices[0]) > 0 else None


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
    
    RESCALED_WIDTH = 1222
    RESCALED_HEIGHT = 859

    def __init__(self, points: np.ndarray, file_path: str, index_mapping: dict = None, joint_points: list[list[int]] = None, preprocessor=LandmarksPreprocessor()) -> None:
        """
        Initializes the Landmarks object.

        Args:
            points (np.ndarray): The array of landmark points.
            file_path (str): The file path associated with the landmarks.
            preprocessor (LandmarksPreprocessor, optional): The preprocessor for the landmarks. Defaults to LandmarksPreprocessor().
        """
        self.__points, mapping = preprocessor.preprocess_points(points)
        self.__points = self.__points.astype(int)
        self.__path = file_path
        
        self.__index_mapping = None
        if index_mapping is not None:
            self.__index_mapping = index_mapping
        elif mapping is not None:
            self.__index_mapping = mapping
        
        if joint_points is not None:
            self.__joint_points = preprocessor.preprocess_joint_points(joint_points)
    
    
    @staticmethod
    def from_vector(vector: np.ndarray, index_mapping: dict = None, input_path: str = None, joint_points: list[list[int]] = None, preprocessor=LandmarksPreprocessor()) -> 'Landmarks':
        """
        Create an Landmarks object from a vector.

        Args:
            vector (np.ndarray): The landmark coordinates as a 1D vector.
            input_path (str, optional): The path of the image file. Defaults to None.
            preprocessor (LandmarksPreprocessor, optional): The landmarks preprocessor. Defaults to LandmarksPreprocessor().

        Returns:
            Landmarks: The Landmarks object.
        """
        assert vector.ndim == 1, 'The input data is not a 1D vector.'
        data = vector.reshape(vector.shape[0] // 2, 2)
        return Landmarks(data, input_path, index_mapping=index_mapping, joint_points=joint_points, preprocessor=preprocessor)

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

    @property
    def joint_points(self) -> list[list[int]]:
        return self.__joint_points
    
    @property
    def index_mapping(self):
        return self.__index_mapping
    
    def show(self, title: str = None, join_points: bool = False) -> None:
        """
        Displays the landmarks.

        Args:
            title (str, optional): The title of the image window. If no title is passed, the path of the image file will be used. Defaults to None.
        """
        background = np.ones((self.RESCALED_HEIGHT, self.RESCALED_WIDTH, 3), dtype=np.uint8) * 255
        
        
        for x, y in self.__points:
            background = cv2.circle(background, (x, y), 2, (255, 0, 0), thickness=-1)
            
        if join_points:
            for line in self.__joint_points:
                for i in range(len(line) - 1):
                    point1_idx = self.__index_mapping[line[i]]
                    point2_idx = self.__index_mapping[line[i + 1]]
                    x1 = self.__points[point1_idx, 0]
                    y1 = self.__points[point1_idx, 1]
                    x2 = self.__points[point2_idx, 0]
                    y2 = self.__points[point2_idx, 1]
                    
                    background = cv2.line(background, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
                    
        title = title if title else self.__path
        cv2.imshow(title, background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
