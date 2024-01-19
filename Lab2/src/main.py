import os
from utils.cfd_loader import CFDLoader
from utils.visualizer import Visualizer
from utils.image import ImagePreprocessor
from utils.landmarks import LandmarksPreprocessor
import numpy as np
from pca import PCA
import cv2


DATA_PATH = 'data'
CFD_DIR = os.path.join(DATA_PATH, 'CFD Version 3.0')
LANDMARKS_DIR = os.path.join(DATA_PATH, 'landmark_templates_01-29.22')


if __name__ == '__main__':
    image_preprocessor = ImagePreprocessor(new_size=(1222, 859), new_color=cv2.COLOR_BGR2GRAY)
    landmarks_preprocessor = LandmarksPreprocessor(new_scale=(1222, 859))
    loader = CFDLoader(CFD_DIR, LANDMARKS_DIR, image_preprocessor, landmarks_preprocessor)
    images = loader.get_images()
    landmarks = loader.get_landmarks()
    print(f'Loaded {len(images)} images and {len(landmarks)} landmarks')

    # # Usage examples. Feel free to delete this.
    # # How to use landmarks
    print(f'Landmarks path: {landmarks[0].path}')
    print(f'Landmarks shape: {landmarks[0].as_matrix().dtype}')
    print(f'Landmarks points:\n{landmarks[0].as_matrix()}\n')
    
    
    image_visualizer = Visualizer(images, landmarks)
    image_visualizer.visualize(show_images=True, show_landmarks=True)
    image_visualizer.show_all_landmarks()
    


    # image_data = np.stack([image.as_vector() for image in images], axis=1)
    # # print(image_data.shape[1])
    # print(image_data.shape)
    # images_pca = PCA(image_data)
    # images_pca.scree_plot()


    # # How to use images
    # for i in range(5):
    #     # print(f'Image path: {images[i].path}')
    #     # print(f'Image shape: {images[i].as_matrix().shape}')
    #     # print(f'Image flattened shape: {images[i].as_vector().shape}')
    #     images[i].show()

    # print(f'Original image pixels:\n{images[0].as_matrix()}\n')
    # print(f'Flattened image pixels:\n{images[0].as_vector()}\n')
    # assert((images[0].as_matrix() == images[0].as_vector().reshape(images[0].height, images[0].width, 3)).all())
