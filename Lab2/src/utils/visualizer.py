from utils.image import Image
from utils.landmarks import Landmarks
import cv2 as cv

class CarrousselManager:
    def __init__(self, num_images):
        self.num_images = num_images
        self.__current_idx = 0
        self.__unkown_key = True
        self.__next_key = False
        self.__prev_key = False
    
    def next(self):
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
    def __init__(self, images: list[Image], landmarks: list[Landmarks]) -> None:
        self.images = images
        self.landmarks = landmarks
        
        
        self.information = {}
        self.__order_information()
        
        self.carroussel_manager = CarrousselManager(len(self.images))
    
    
    def __order_information(self):
        for image in self.images:
            image_coords = image.as_matrix()
            landmark_coords = self.__find_landmark(image.path.split("/")[-1])
            self.information[image.path] = {"image_coords": image_coords, "landmark_coords": landmark_coords}

    
    def __find_landmark(self, file_name: str):
        for landmark in self.landmarks:
            if landmark.path.split(".")[0] == file_name.split(".")[0]:
                return landmark.as_matrix()
    
    
    def visualize(self, show_images=True, show_landmarks=True):
        curr_idx = self.carroussel_manager.next()
        
        #TODO: add functionality to optionally display the landmarks or the images
        while True:
            curr_image = self.images[curr_idx]
            curr_landmarks = self.information[curr_image.path]["landmark_coords"]
            
            image_coords = curr_image.as_matrix().copy()
            
            for x, y in curr_landmarks:
                image_coords = cv.circle(image_coords, (int(x), int(y)), 2, (255, 0, 0), thickness=4)
                

            
            cv.imshow(curr_image.path, image_coords)
            
            key_pressed = cv.waitKey(0)
            if key_pressed == 27:
                cv.destroyAllWindows()
                break
            elif key_pressed == 100: # 'D'
                curr_idx = self.carroussel_manager.next()
            elif key_pressed == 97: # 'A'
                curr_idx = self.carroussel_manager.prev()
            cv.destroyAllWindows()