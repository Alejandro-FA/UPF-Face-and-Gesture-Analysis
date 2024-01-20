import os
from eigenfaces.utils.visualizer import Visualizer
from eigenfaces.utils.image import Image
from precomputations import load_precomputations


RESULTS_PATH = 'assets'
DATA_PATH = 'data'
PICKLES_PATH = 'pickles'
DOWNSAMPLE_SIZE = (1222, 859)
COMPUTE_SCREE_PLOT = True


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
    p = 30
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

    # Visualize the first p eigenvectors
    eigenvectors = images_pca.eigenvectors
    eigenvalues = images_pca.eigenvalues
    for i in range(p):
        e = eigenvectors[:, i] * eigenvalues[i] * 255
        image = Image.from_vector(e, DOWNSAMPLE_SIZE)
        # print(image.as_matrix())
        image.show(title=f'Eigenvector {i}')
        
    # Visualize the last p eigenvectors
    eigenvectors = images_pca.eigenvectors
    for i in range(1, p + 1):
        e = eigenvectors[:, -i - 100] * eigenvalues[-i - 100] * 255
        image = Image.from_vector(e, DOWNSAMPLE_SIZE)
        image.show(title=f'Eigenvector {len(e) - i}')


    # mean_face + eigenvector * 1*sigma
        
    

    # eigenvectors = [
    #     Image.from_vector(eigenvectors[:, i], DOWNSAMPLE_SIZE)
    #     for i in range(eigenvectors.shape[1])
    # ]
    # eigenvectors_visualizer = Visualizer(eigenvectors, landmarks)
    # eigenvectors_visualizer.visualize(show_landmarks=False)
    
    # # Show plots
    # plt.show()
