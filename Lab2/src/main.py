import os
import matplotlib.pyplot as plt
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

    # Data exploration
    image_visualizer = Visualizer(images, landmarks)
    image_visualizer.visualize(show_images=True, show_landmarks=True)
    image_visualizer.show_all_landmarks()

    # PCA results evaluation
    fig = images_pca.scree_plot()
    fig.savefig(os.path.join(RESULTS_PATH, 'scree_plot.png'))

    # Principal components
    p = 10
    image_data = images_pca.data
    print('Computing principal components...')
    pcs = images_pca.to_pca_space(image_data, num_components=p)
    print('Reconstruing images to the original space...')
    reconstruction = images_pca.from_pca_space(pcs)
    print('Done!')

    for i, image in enumerate(images[0:5]):
        reconstructed_image = Image.from_vector(reconstruction[:, i], DOWNSAMPLE_SIZE)
        reconstructed_image.show()

    # Show mean face and first 10 principal components and the full reconstruction
    mean_face = Image.from_vector(images_pca.mean, DOWNSAMPLE_SIZE)
    mean_face.show('Mean face')

    # Show plots
    plt.show()
