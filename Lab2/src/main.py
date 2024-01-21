import os
from eigenfaces.utils.visualizer import Visualizer
from eigenfaces.utils.image import Image
from eigenfaces.utils.landmarks import Landmarks
from precomputations import load_precomputations
import numpy as np
import imageio
import cv2



RESULTS_PATH = 'assets'
DATA_PATH = 'data'
PICKLES_PATH = 'pickles'
DOWNSAMPLE_SIZE = (1222, 859)
COMPUTE_SCREE_PLOT = False


if __name__ == '__main__':
    # Load data
    precomputations = load_precomputations(DATA_PATH, PICKLES_PATH)
    images = precomputations.images
    landmarks = precomputations.landmarks
    images_pca = precomputations.images_pca
    landmarks_pca = precomputations.landmarks_pca

    # Compute scree plot
    if COMPUTE_SCREE_PLOT:
        print('\nComputing scree plot...')
        fig = landmarks_pca.scree_plot(max_eigenvalues=20, num_permutations=100) # WARNING: This takes a long time to compute!
        fig.savefig(os.path.join(RESULTS_PATH, 'landmarks_scree_plot.png'), dpi=1000)
        print('Done!')
        exit()

    # # Data exploration
    image_visualizer = Visualizer(images, landmarks)
    image_visualizer.visualize(show_images=True, show_landmarks=True, show_landmarks_idx=True)
    # image_visualizer.show_all_landmarks()

    # Compute principal components
    p = 10
    image_data = images_pca.data
    landmark_data = landmarks_pca.data
    
    print('\nComputing principal components...')
    pcs_image = images_pca.to_pca_space(image_data, num_components=p)
    pcs_data = landmarks_pca.to_pca_space(landmark_data, num_components=p)
    print('Reconstructing images to the original space...')
    reconstruction_image = images_pca.from_pca_space(pcs_image)
    reconstruction_landmarks = landmarks_pca.from_pca_space(pcs_data)
    print('Done!')
    
    # # Visualize principal components
    # reconstructed_images = [
    #     Image.from_vector(reconstruction_image[:, i], DOWNSAMPLE_SIZE, input_path=image.path)
    #     for i, image in enumerate(images)
    # ]
    
    # reconstructed_landmarks = [
    #     Landmarks.from_vector(reconstruction_landmarks[:, i], input_path=landmark.path)
    #     for i, landmark in enumerate(landmarks)
    # ]
    # print(reconstructed_landmarks[0].path)
    # pca_visualizer = Visualizer(reconstructed_images, reconstructed_landmarks)
    # pca_visualizer.visualize(show_landmarks=True)

    # Show mean face
    mean_face = Image.from_vector(images_pca.mean, DOWNSAMPLE_SIZE)
    mean_face.show(title='Mean face')
    
    
    # Show mean landmarks
    print(landmarks[0].index_mapping)
    mean_landmarks = Landmarks.from_vector(landmarks_pca.mean, index_mapping=landmarks[0].index_mapping, joint_points=landmarks[0].joint_points)
    mean_landmarks.show(title="Mean landmarks", join_points=True)
    

    eigenvectors_images = images_pca.eigenvectors
    eigenvalues_images = images_pca.eigenvalues
    
    eigenvectors_landmarks = landmarks_pca.eigenvectors
    eigenvalues_landmarks = landmarks_pca.eigenvalues
    
    
    # for base_num in range(15):
    #     base_path = f"assets/eigenvector_{base_num}/"
    #     if not os.path.exists(base_path):
    #         os.makedirs(base_path)
    #     std = np.sqrt(np.sqrt(eigenvalues[base_num]))
        
    #     width = images[0].width
    #     height = images[0].height
    #     output_imgs = []
        
    #     for idx, i in enumerate(np.arange(-10 * std, 10 * std, 20 * std / 30)):
    #         varied_face = mean_face.as_vector() + eigenvectors[:, base_num] * i * 255
    #         varied_face_img = Image.from_vector(varied_face, DOWNSAMPLE_SIZE, input_path=images[base_num].path)
    #         cv2.imwrite(f"{base_path}{idx}.png", varied_face_img.as_matrix())
    
    

    # eigenvectors = [
    #     Image.from_vector(eigenvectors_images[:, i] * eigenvalues_images[i] * 255, DOWNSAMPLE_SIZE, images[i].path)
    #     for i in range(eigenvectors_images.shape[1])
    # ]
    # eigenvectors_img_visualizer = Visualizer(eigenvectors, landmarks)
    # eigenvectors_img_visualizer.visualize(show_landmarks=False)
    
    # print(Landmarks.from_vector(eigenvectors_landmarks[:, 0] * eigenvalues_landmarks[0]).as_matrix())
    # print(Landmarks.from_vector(eigenvectors_landmarks[:, 0] * eigenvalues_landmarks[0]).as_matrix() + np.array([859, 1222]))
    # print(Landmarks.from_vector(eigenvectors_landmarks[:, 0] * eigenvalues_landmarks[0]).as_matrix().shape)
    
    # eigenvectors = [
    #     Landmarks.from_vector(eigenvectors_landmarks[:, i] * eigenvalues_landmarks[i], landmarks[i].path)
    #     for i in range(eigenvectors_landmarks.shape[1])
    # ]
    # eigenvectors_img_visualizer = Visualizer(eigenvectors, landmarks)
    # eigenvectors_img_visualizer.visualize(show_landmarks=False)
    
    # # Show plots
    # plt.show()
