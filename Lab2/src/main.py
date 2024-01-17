import os
from utils.cfd_loader import CFDLoader

DATA_PATH = 'data'
CFD_DIR = os.path.join(DATA_PATH, 'CFD Version 3.0')
LANDMARKS_DIR = os.path.join(DATA_PATH, 'landmark_templates_01-29.22')


if __name__ == '__main__':
    loader = CFDLoader(CFD_DIR, LANDMARKS_DIR)
    images = loader.get_images()
    landmarks = loader.get_landmarks()
    print(f'Loaded {len(images)} images and {len(landmarks)} landmarks')

    # Usage examples. Feel free to delete this.
    # How to use landmarks
    print(f'Landmarks path: {landmarks[0].path}')
    print(f'Landmarks shape: {landmarks[0].as_matrix().shape}')
    print(f'Landmarks points:\n{landmarks[0].as_matrix()}\n')

    # How to use images
    for i in range(5):
        print(f'Image path: {images[i].path}')
        print(f'Image shape: {images[i].as_matrix().shape}')
        print(f'Image flattened shape: {images[i].as_vector().shape}')
        images[i].show()

    print(f'Original image pixels:\n{images[0].as_matrix()}\n')
    print(f'Flattened image pixels:\n{images[0].as_vector()}\n')
    assert((images[0].as_matrix() == images[0].as_vector().reshape(images[0].height, images[0].width, 3)).all())
