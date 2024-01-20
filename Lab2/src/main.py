import os
from eigenfaces.utils.visualizer import Visualizer
from eigenfaces.utils.image import Image
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

    # Compute scree plot
    if COMPUTE_SCREE_PLOT:
        print('\nComputing scree plot...')
        fig = images_pca.scree_plot(max_eigenvalues=20, num_permutations=100) # WARNING: This takes a long time to compute!
        fig.savefig(os.path.join(RESULTS_PATH, 'scree_plot.png'), dpi=1000)
        print('Done!')
        exit()

    # # Data exploration
    # image_visualizer = Visualizer(images, landmarks)
    # image_visualizer.visualize(show_images=True, show_landmarks=True)
    # image_visualizer.show_all_landmarks()

    # Compute principal components
    p = 10
    image_data = images_pca.data
    
    print('\nComputing principal components...')
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
    mean_face.show(title='Mean face')

    eigenvectors = images_pca.eigenvectors
    eigenvalues = images_pca.eigenvalues
    
    # # Visualize the first p eigenvectors
    # for i in range(p):
    #     e = eigenvectors[:, i] * eigenvalues[i] * 255
    #     image = Image.from_vector(e, DOWNSAMPLE_SIZE)
    #     # print(image.as_matrix())
    #     image.show(title=f'Eigenvector {i}')
        
    # # Visualize the last p eigenvectors
    # eigenvectors = images_pca.eigenvectors
    # for i in range(1, p + 1):
    #     e = eigenvectors[:, -i - 100] * eigenvalues[-i - 100] * 255
    #     image = Image.from_vector(e, DOWNSAMPLE_SIZE)
    #     image.show(title=f'Eigenvector {len(e) - i}')


    
    
    for base_num in range(15):
        base_path = f"assets/eigenvector_{base_num}/"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        std = np.sqrt(np.sqrt(eigenvalues[base_num]))
        
        width = images[0].width
        height = images[0].height
        output_imgs = []
        
        for idx, i in enumerate(np.arange(-10 * std, 10 * std, 20 * std / 30)):
            varied_face = mean_face.as_vector() + eigenvectors[:, base_num] * i * 255
            varied_face_img = Image.from_vector(varied_face, DOWNSAMPLE_SIZE, input_path=images[base_num].path)
            cv2.imwrite(f"{base_path}{idx}.png", varied_face_img.as_matrix())
    
    

    # eigenvectors = [
    #     Image.from_vector(eigenvectors[:, i], DOWNSAMPLE_SIZE)
    #     for i in range(eigenvectors.shape[1])
    # ]
    # eigenvectors_visualizer = Visualizer(eigenvectors, landmarks)
    # eigenvectors_visualizer.visualize(show_landmarks=False)
    
    # # Show plots
    # plt.show()
