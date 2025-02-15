import cv2
import numpy as np
from .image import Image
from .landmarks import Landmarks

class CarrousselManager:
    """
    A class for managing the carousel of images.

    Attributes:
        num_images (int): The total number of images in the carousel.
        __current_idx (int): The current index of the selected image.
        __unknown_key (bool): Flag indicating an unknown key state.
        __next_key (bool): Flag indicating the next key state.
        __prev_key (bool): Flag indicating the previous key state.
    """
    def __init__(self, num_images):
        """
        Initializes a CarrousselManager object with the total number of images.

        Parameters:
            num_images (int): The total number of images in the carousel.
        """
        self.num_images = num_images
        self.__current_idx = 0
        self.__unkown_key = True
        self.__next_key = False
        self.__prev_key = False
    
    def next(self):
        """
        Returns the index of the next image in the carousel.

        Returns:
            int: The index of the next image.
        """
        if self.__unkown_key:
            select_idx = self.__current_idx
            self.__current_idx += 1
            self.__unkown_key = False
            self.__next_key = True
        elif self.__next_key:
            select_idx = self.__current_idx
            self.__current_idx += 1
        elif self.__prev_key:
            select_idx = (self.__current_idx + 2) % self.num_images
            self.__current_idx += 3
            self.__prev_key = False
            self.__next_key = True
        
        self.__current_idx = self.__current_idx % self.num_images
        
        return select_idx
    
    def prev(self):
        """
        Returns the index of the previous image in the carousel.

        Returns:
            int: The index of the previous image.
        """
        if self.__unkown_key:
            select_idx = self.__current_idx
            self.__current_idx -= 1
            self.__unkown_key = False
            self.__prev_key = True
        elif self.__next_key:
            select_idx = (self.__current_idx - 2) % self.num_images
            self.__current_idx -= 3
            self.__next_key = False
            self.__prev_key = True
        elif self.__prev_key:
            select_idx = self.__current_idx
            self.__current_idx -= 1
        
        self.__current_idx = self.__current_idx % self.num_images

        return select_idx


class Visualizer:
    """
    A class for visualizing images with associated landmarks.

    Attributes:
        images (list[Image]): A list of Image objects representing the images to be visualized.
        landmarks (list[Landmarks]): A list of Landmarks objects representing the landmarks associated with the images.
        information (dict): A dictionary containing ordered information about image coordinates and landmark coordinates.
    """
    IMAGES_WIDTH = 1222
    IMAGES_HEIGHT = 859
    
    def __init__(self, images: list[Image], landmarks: list[Landmarks]) -> None:
        """
        Initializes a Visualizer object with a list of images and landmarks.

        Parameters:
            images (list[Image]): A list of Image objects representing the images to be visualized.
            landmarks (list[Landmarks]): A list of Landmarks objects representing the landmarks associated with the images.

        """
        self.images = images
        self.landmarks = landmarks
        
        
        self.data = {}
        self.__match_landmarks_images()
        
        self.carroussel_manager = CarrousselManager(len(self.images))
    
    
    def __match_landmarks_images(self):
        """
        Matches the images with their corresponding landmark coordinates
        """
        for image in self.images:
            image_coords = image.as_matrix()
            landmark_coords = self.__find_landmark(image.path.split("/")[-1])
            self.data[image.path] = {"image_coords": image_coords, "landmark_coords": landmark_coords}

    
    def __find_landmark(self, file_name: str):
        """
        Finds landmark coordinates corresponding to a given file name.

        Parameters:
            file_name (str): The file name for which to find landmark coordinates.

        Returns:
            list: A list of landmark coordinates.
        """
        
        for landmark in self.landmarks:
            if landmark.path.split(".")[0] == file_name.split(".")[0]:
                return landmark.as_matrix()
    
    
    def visualize(self, show_images=True, show_landmarks=True, show_landmarks_idx=False):
        """
        Visualizes images with optional display of landmarks.

        Parameters:
            show_images (bool): Whether to display images
            show_landmarks (bool): Whether to display landmarks
            show_landmarks_idx (bool): Whether to display landmark indices
        """
        
        curr_idx = self.carroussel_manager.next()
        processed_images = [None] * len(self.images)
        
        while True:
            if processed_images[curr_idx] is None:
                curr_image = self.images[curr_idx]
                curr_landmarks = self.data[curr_image.path]["landmark_coords"]
                
                if show_images:
                    image_coords = curr_image.as_matrix().copy()
                else:
                    image_coords = np.ones((self.IMAGES_HEIGHT, self.IMAGES_WIDTH, 3), dtype=np.uint8)
                    
                
                if show_landmarks:
                    for i, (x, y) in enumerate(curr_landmarks):
                        image_coords = cv2.circle(image_coords, (x, y), 2, (0, 0, 0), thickness=-1)
                        if show_landmarks_idx:
                            image_coords = cv2.putText(image_coords, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1, cv2.LINE_AA)
                
                processed_images[curr_idx] = image_coords
            else:
                image_coords = processed_images[curr_idx]

            
            cv2.imshow(curr_image.path, image_coords)
            
            key_pressed = cv2.waitKey(0)
            if key_pressed == 27:
                cv2.destroyAllWindows()
                break
            elif key_pressed == 100: # 'D'
                curr_idx = self.carroussel_manager.next()
            elif key_pressed == 97: # 'A'
                curr_idx = self.carroussel_manager.prev()
            cv2.destroyAllWindows()
    
    def show_all_landmarks(self):
        """
        Displays all landmarks on a blank background.
        """
        background = np.ones((self.IMAGES_HEIGHT, self.IMAGES_WIDTH, 3), dtype=np.uint8) * 255
        
        for landmark in self.landmarks:
            for x, y in landmark.as_matrix():
                background = cv2.circle(background, (x, y), 2, (255, 0, 0), thickness=-1)
        
        cv2.imshow("All landmarks", background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
