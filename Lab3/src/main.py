import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from realiability import reliability
from mds import MDS
from utils import dissimilarity_matrix

if __name__ == '__main__':
    data = loadmat("output.mat")

    # FIXME: use np.sequeeze or more elegant alternative to load the data
    fake = np.random.randint(0, 9, size=(24, 24))
    similarity_matrix: np.ndarray = data["simScores"][0][0][0]
    consistency_matrix: np.ndarray = data["simScores"][0][0][1]

    error, p_value = reliability(similarity_matrix, consistency_matrix)
    print(f'P value: {p_value}')
    print(f'Error: {error}')

    # Compute MDS
    dissimilar_matrix = dissimilarity_matrix(similarity_matrix)
    mds = MDS(dissimilar_matrix)
    eigenvalues, eigenvectors = mds.eigenvalues, mds.eigenvectors
    
    # print(f'Eigenvalues: {eigenvalues}')
    # print(f'Eigenvectors: {eigenvectors}')

    # fig = mds.scree_plot(num_resamples=1000)
    # plt.show()
    emotions = ["angry", "boredom", "disgusted", "friendly", "happiness", "laughter", "sadness", "surprised"]
    colors = ["red", "gray", "green", "pink", "yellow", "cyan", "black", "orange"]
    
    mds_distances = mds.to_mds_space(2)
    
    for i in np.arange(0, 24, 3):
        plt.scatter(mds_distances[0, i:i+3], mds_distances[1, i:i+3], color=colors[i // 3], label=emotions[i // 3])
        
    plt.legend()
    plt.show()