import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.visualizer import Visualizer
from utils.image import Image
from precomputations import load_precomputations


RESULTS_PATH = 'assets'
DATA_PATH = 'data'
DOWNSAMPLE_SIZE = (1222, 859)


if __name__ == '__main__':
    # Load data
    precomputations = load_precomputations(DATA_PATH)
    images = precomputations.images
    landmarks = precomputations.landmarks
    images_pca = precomputations.images_pca

    # # Data exploration
    # image_visualizer = Visualizer(images, landmarks)
    # image_visualizer.visualize(show_images=True, show_landmarks=True)
    # image_visualizer.show_all_landmarks()

    # PCA results evaluation
    fig = images_pca.scree_plot()
    fig.savefig(os.path.join(RESULTS_PATH, 'scree_plot.png'))

    # Compute principal components
    p = 500
    image_data = images_pca.data
    
    print('Computing principal components...')
    pcs = images_pca.to_pca_space(image_data, num_components=p)
    print('Reconstructing images to the original space...')
    reconstruction = images_pca.from_pca_space(pcs)
    print('Done!')
    
    # Visualize principal components
    reconstructed_images = [
        Image.from_vector(reconstruction[:, i], DOWNSAMPLE_SIZE, input_path=image.path)
        for i, image in enumerate(images)
    ]
    pca_visualizer = Visualizer(reconstructed_images, landmarks)
    pca_visualizer.visualize(show_landmarks=False)

    # Show mean face
    mean_face = Image.from_vector(images_pca.mean, DOWNSAMPLE_SIZE)
    mean_face.show('Mean face')

    # # Show plots
    # plt.show()
