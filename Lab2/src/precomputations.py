import os
import numpy as np
import cv2
import pickle
from typing import NamedTuple
from eigenfaces.utils.cfd_loader import CFDLoader
from eigenfaces.utils.image import Image, ImagePreprocessor
from eigenfaces.utils.landmarks import Landmarks, LandmarksPreprocessor
from eigenfaces.pca import PCA


DOWNSAMPLE_SIZE = (1222, 859)
IMAGES_FILE = 'images.pkl'
LANDMARKS_FILE = 'landmarks.pkl'
IMAGES_PCA_FILE = 'images_pca.pkl'
LANDMARKS_PCA_FILE = 'landmarks_pca.pkl'


class Precomputations(NamedTuple):
    images: list[Image]
    landmarks: list[Landmarks]
    images_pca: PCA
    landmarks_pca: PCA


def load_precomputations(data_path: str, pickles_path: str) -> Precomputations:
    """
    Load precomputed data from files and return a Precomputations object.

    Args:
        data_path (str): The path to the directory containing the precomputed data files.

    Returns:
        Precomputations: An object containing the loaded precomputed data.

    Raises:
        FileNotFoundError: If the precomputed data files are not found, the function will create them and recursively call itself to load the data.
    """
    try:
        with open(os.path.join(pickles_path, IMAGES_FILE), 'rb') as f:
            images = pickle.load(f)
        with open(os.path.join(pickles_path, LANDMARKS_FILE), 'rb') as f:
            landmarks = pickle.load(f)
        with open(os.path.join(pickles_path, IMAGES_PCA_FILE), 'rb') as f:
            images_pca = pickle.load(f)
        with open(os.path.join(pickles_path, LANDMARKS_PCA_FILE), 'rb') as f:
            landmarks_pca = pickle.load(f)
            
        print('\nPrecomputations loaded')
        return Precomputations(images, landmarks, images_pca, landmarks_pca)
    
    except FileNotFoundError:
        print('\n---------------------------------------------')
        print('Precomputations not found, creating them...\n')
        __do_precomputations(data_path, pickles_path)
        print('\nPrecomputations created!')
        print('---------------------------------------------')
        return load_precomputations(data_path, pickles_path)


def __do_precomputations(data_path: str, pickles_path: str) -> None:
    # Create pickles directory if it doesn't exist
    if not os.path.exists(pickles_path):
        os.makedirs(pickles_path)

    # Save loaded data
    images, landmarks = __load_data(data_path)
    with open(os.path.join(pickles_path, IMAGES_FILE), 'wb') as f:
        pickle.dump(images, f)
    with open(os.path.join(pickles_path, LANDMARKS_FILE), 'wb') as f:
        pickle.dump(landmarks, f)
    
    # Compute PCA
    pca_result_images = __compute_pca(images)
    with open(os.path.join(pickles_path, IMAGES_PCA_FILE), 'wb') as f:
        pickle.dump(pca_result_images, f)
    print("PCA for images completed")
    
    pca_result_landmarks = __compute_pca(landmarks)
    with open(os.path.join(pickles_path, LANDMARKS_PCA_FILE), 'wb') as f:
        pickle.dump(pca_result_landmarks, f)
    print("PCA for landmarks completed")
    
    

def __load_data(data_path: str) -> tuple[list[Image], list[Landmarks]]:
    # Paths
    cfd_dir = os.path.join(data_path, 'CFD Version 3.0')
    landmarks_dir = os.path.join(data_path, 'landmark_templates_01-29.22')

    # Create preprocessors
    p = list(range(145, 158)) + [183, 184] + list(range(135, 144))
    landmarks_preprocessor = LandmarksPreprocessor(new_scale=DOWNSAMPLE_SIZE, unwanted_points=p)
    image_preprocessor = ImagePreprocessor(new_size=DOWNSAMPLE_SIZE, new_color=cv2.COLOR_BGR2GRAY)

    # Load data
    loader = CFDLoader(cfd_dir, landmarks_dir, image_preprocessor, landmarks_preprocessor)
    images = loader.get_images()
    landmarks = loader.get_landmarks()
    print(f'Loaded {len(images)} images and {len(landmarks)} landmarks')

    return images, landmarks


def __compute_pca(data: list[Image | Landmarks]) -> PCA:
    print('\nCreating data matrix...')
    vectorized_data = np.stack([val.as_vector() for val in data], axis=1)
    print('Computing eigendecomposition...')
    pca_result = PCA(vectorized_data)
    print('Done!')

    return pca_result
