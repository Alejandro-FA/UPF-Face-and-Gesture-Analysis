import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.cfd_loader import CFDLoader
from utils.visualizer import Visualizer
from utils.image import Image, ImagePreprocessor
from utils.landmarks import LandmarksPreprocessor
from pca import PCA


RESULTS_PATH = 'assets'
DATA_PATH = 'data'
CFD_DIR = os.path.join(DATA_PATH, 'CFD Version 3.0')
LANDMARKS_DIR = os.path.join(DATA_PATH, 'landmark_templates_01-29.22')

DOWNSAMPLE_SIZE = (1222, 859)


if __name__ == '__main__':
    # Load data
    p = list(range(145, 158)) + [183, 184] + list(range(135, 144))
    landmarks_preprocessor = LandmarksPreprocessor(new_scale=DOWNSAMPLE_SIZE, unwanted_points=p)
    image_preprocessor = ImagePreprocessor(new_size=DOWNSAMPLE_SIZE, new_color=cv2.COLOR_BGR2GRAY)

    loader = CFDLoader(CFD_DIR, LANDMARKS_DIR, image_preprocessor, landmarks_preprocessor)
    images = loader.get_images()
    landmarks = loader.get_landmarks()
    print(f'Loaded {len(images)} images and {len(landmarks)} landmarks')

    # Data exploration
    image_visualizer = Visualizer(images, landmarks)
    image_visualizer.visualize(show_images=True, show_landmarks=True)
    image_visualizer.show_all_landmarks()
    
    # PCA
    print('\nCreating data matrix...')
    image_data = np.stack([image.as_vector() for image in images], axis=1)
    print('Computing eigendecomposition...')
    images_pca = PCA(image_data)
    print('Done!')

    # PCA results evaluation
    fig = images_pca.scree_plot()
    fig.savefig(os.path.join(RESULTS_PATH, 'scree_plot.png'))

    # Principal components
    p = 10
    print('\nComputing principal components...')
    pcs = images_pca.to_pca_space(image_data, num_components=p)
    print('Reconstruing images to the original space...')
    reconstruction = images_pca.from_pca_space(pcs, num_components=p)
    print('Done!')

    for i, image in enumerate(images[0:5]):
        reconstructed_image = Image.from_vector(reconstruction[:, i], DOWNSAMPLE_SIZE)
        reconstructed_image.show()

    # Show mean face and first 10 principal components and the full reconstruction
    # TODO:

    # Show plots
    plt.show()
