import os
from eigenfaces.utils.visualizer import Visualizer
from eigenfaces.utils.image import Image
from eigenfaces.utils.landmarks import Landmarks
from precomputations import load_precomputations
import numpy as np
import cv2
import matplotlib.pyplot as plt



RESULTS_PATH = 'assets'
DATA_PATH = 'data'
PICKLES_PATH = 'pickles'
DOWNSAMPLE_SIZE = (1222, 859)

DO_DATA_EXPLORATION = False
COMPUTE_IMAGE_SCREE_PLOT = False
COMPUTE_LANDMARKS_SCREE_PLOT = True


if __name__ == '__main__':
    ###########################################################################
    # Load data and precomputations
    ###########################################################################
    precomputations = load_precomputations(DATA_PATH, PICKLES_PATH)
    images = precomputations.images
    landmarks = precomputations.landmarks
    images_pca = precomputations.images_pca
    landmarks_pca = precomputations.landmarks_pca

    # Data exploration
    if DO_DATA_EXPLORATION:
        print('Data exploration...')
        image_visualizer = Visualizer(images, landmarks)
        image_visualizer.visualize(show_images=True, show_landmarks=True, show_landmarks_idx=True)
        image_visualizer.show_all_landmarks()


    ###########################################################################
    # Face appearance model (images)
    ###########################################################################
    print('\n---------------------------------------------')
    print('Face appearance model (images)')

    # Compute scree plot
    if COMPUTE_IMAGE_SCREE_PLOT:
        print('\nComputing scree plot...')
        fig = images_pca.scree_plot(max_eigenvalues=20, num_permutations=100) # WARNING: This takes a long time to compute!
        fig.savefig(os.path.join(RESULTS_PATH, 'images_scree_plot.png'), dpi=1000)
        print('Done!')
        plt.show()

    # Compute principal components
    p = 10
    print('\nComputing principal components...')
    pcs_image = images_pca.to_pca_space(images_pca.data, num_components=p)
    print('Reconstructing images to the original space...')
    reconstruction_image = images_pca.from_pca_space(pcs_image)
    print('Done! Visualizing reconstructed images...')
    
    # Visualize principal components
    reconstructed_images = [
        Image.from_vector(reconstruction_image[:, i], DOWNSAMPLE_SIZE, input_path=image.path)
        for i, image in enumerate(images)
    ]
    for image in reconstructed_images[0:5]:
        image.show()

    # Show mean face
    print('\nVisualizing mean face...')
    mean_face = Image.from_vector(images_pca.mean, DOWNSAMPLE_SIZE)
    mean_face.show(title='Mean face')
    
    # Modes of variation of the first 15 eigenvectors
    print('\nComputing modes of variation of the first 15 eigenvectors...')
    eigenvectors_images = images_pca.eigenvectors
    eigenvalues_images = images_pca.eigenvalues
    
    for base_num in range(15):
        base_path = os.path.join(RESULTS_PATH, f'image_eigenvector_{base_num}')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        std = np.sqrt(eigenvalues_images[base_num])
        
        width = images[0].width
        height = images[0].height
        output_imgs = []
        
        for idx, a in enumerate(np.linspace(-3 * std, 3 * std, 30)):
            varied_face = mean_face.as_vector() + eigenvectors_images[:, base_num] * a
            varied_face_img = Image.from_vector(varied_face, DOWNSAMPLE_SIZE, input_path=images[base_num].path)
            # varied_face_img.show()
            file_path = os.path.join(base_path, f'{idx}.png')
            cv2.imwrite(file_path, varied_face_img.as_matrix()) # FIXME: implement this as a method in Image class
    
    print('---------------------------------------------')
    
    ###########################################################################
    # Point distribution model (landmarks)
    ###########################################################################
    print('\n---------------------------------------------')
    print('Point distribution model (landmarks)')

    # Compute scree plot
    if COMPUTE_LANDMARKS_SCREE_PLOT:
        print('\nComputing scree plot...')
        fig = landmarks_pca.scree_plot(max_eigenvalues=20, num_permutations=100)
        fig.savefig(os.path.join(RESULTS_PATH, 'landmarks_scree_plot.png'), dpi=1000)
        print('Done!')
        plt.show()

    # Compute principal components
    p = 10
    print('\nComputing principal components...')
    pcs_data = landmarks_pca.to_pca_space(landmarks_pca.data, num_components=p)
    print('Reconstructing images to the original space...')
    reconstruction_landmarks = landmarks_pca.from_pca_space(pcs_data)
    print('Done! Visualizing reconstructed landmarks...')

    # Visualize principal components
    reconstructed_landmarks = [
        Landmarks.from_vector(reconstruction_landmarks[:, i], input_path=landmark.path)
        for i, landmark in enumerate(landmarks)
    ]
    for landmark in reconstructed_landmarks[0:5]:
        landmark.show()
            
    # Show mean landmarks
    print('\nVisualizing mean landmarks...')
    mean_landmarks = Landmarks.from_vector(landmarks_pca.mean, index_mapping=landmarks[0].index_mapping, joint_points=landmarks[0].joint_points)
    mean_landmarks.show(title="Mean landmarks", join_points=True)

    # Modes of variation of the first 15 eigenvectors
    print('\nComputing modes of variation of the first 15 eigenvectors...')
    eigenvectors_landmarks = landmarks_pca.eigenvectors
    eigenvalues_landmarks = landmarks_pca.eigenvalues
    
    for base_num in range(15):
        base_path = os.path.join(RESULTS_PATH, f'landmarks_eigenvector_{base_num}')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        std = np.sqrt(eigenvalues_landmarks[base_num])
        
        width = images[0].width
        height = images[0].height
        output_imgs = []
        
        for idx, a in enumerate(np.linspace(-3 * std, 3 * std, 30)):
            varied_face = mean_landmarks.as_vector() + eigenvectors_landmarks[:, base_num] * a
            varied_face_img = Landmarks.from_vector(varied_face, index_mapping=landmarks[0].index_mapping, joint_points=landmarks[0].joint_points, input_path=images[base_num].path)
            file_path = os.path.join(base_path, f'{idx}.png')
            cv2.imwrite(file_path, varied_face_img.get_as_image(join_points=True))

    print('---------------------------------------------')

    ###########################################################################
    # Visualize eigenvectors directly # FIXME: perhaps remove this
    ###########################################################################
    # eigenvectors = [
    #     Image.from_vector(eigenvectors_images[:, i] * eigenvalues_images[i] * 255, DOWNSAMPLE_SIZE, images[i].path)
    #     for i in range(eigenvectors_images.shape[1])
    # ]
    # eigenvectors_img_visualizer = Visualizer(eigenvectors, landmarks)
    # eigenvectors_img_visualizer.visualize(show_landmarks=False)
    
    # eigenvectors = [
    #     Landmarks.from_vector(eigenvectors_landmarks[:, i] * eigenvalues_landmarks[i], landmarks[i].path)
    #     for i in range(eigenvectors_landmarks.shape[1])
    # ]
    # eigenvectors_img_visualizer = Visualizer(eigenvectors, landmarks)
    # eigenvectors_img_visualizer.visualize(show_landmarks=False)
