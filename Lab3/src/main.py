import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from realiability import reliability
from mds import MDS
from utils import dissimilarity_matrix, visualize_similarity_matrix

RESULTS_FOLDER = 'assets'

if __name__ == '__main__':
    data = loadmat("output.mat")
    emotions = ["angry", "boredom", "disgusted", "friendly", "happiness", "laughter", "sadness", "surprised"]
    colors = ["red", "gray", "green", "pink", "yellow", "cyan", "black", "orange"]

    # FIXME: use np.sequeeze or more elegant alternative to load the data
    fake = np.random.randint(0, 9, size=(24, 24))
    similarity_matrix: np.ndarray = data["simScores"][0][0][0]
    consistency_matrix: np.ndarray = data["simScores"][0][0][1]
    
    sim_matrix_fig = visualize_similarity_matrix(similarity_matrix, emotions)
    sim_matrix_fig.savefig(f"{RESULTS_FOLDER}/sim_matrix.png", dpi=500)

    error, p_value = reliability(similarity_matrix, consistency_matrix)
    print(f'P value: {p_value}')
    print(f'Error: {error}')

    # Compute MDS
    dissimilar_matrix = dissimilarity_matrix(similarity_matrix)
    mds = MDS(dissimilar_matrix)
    eigenvalues, eigenvectors = mds.eigenvalues, mds.eigenvectors
    
    # print(f'Eigenvalues: {eigenvalues}')
    # print(f'Eigenvectors: {eigenvectors}')

    # Compute plots
    # scree_plot = mds.scree_plot(num_resamples=1000)
    # scree_plot.savefig(f"{RESULTS_FOLDER}/scree_plot.png", dpi=500)
    
    
    circumplex_model_plot = mds.circumplex_model_plot(colors=colors, emotions=emotions, flip_x=True)
    circumplex_model_plot.savefig(f"{RESULTS_FOLDER}/circumplex_model_plot.png", dpi=500)

    dist_dist_plot_2 = mds.distance_distance_plot(p=2)
    dist_dist_plot_2.savefig(f"{RESULTS_FOLDER}/distance_distance_plot_2.png", dpi=500)
    dist_dist_plot_all = mds.distance_distance_plot()
    dist_dist_plot_all.savefig(f"{RESULTS_FOLDER}/distance_distance_plot_all.png", dpi=500)
    
    plt.show()