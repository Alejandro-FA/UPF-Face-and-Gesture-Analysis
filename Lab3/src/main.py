import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from realiability import reliability
from mds import MDS

if __name__ == '__main__':
    data = loadmat("output.mat")

    # FIXME: use np.sequeeze or more elegant alternative to load the data
    fake = np.random.randint(0, 9, size=(24, 24))
    similarity_matrix: np.ndarray = data["simScores"][0][0][0]
    consistency_matrix: np.ndarray = data["simScores"][0][0][1]

    error, p_value = reliability(similarity_matrix, consistency_matrix)
    print(f'P value: {p_value}')
    print(f'Error: {error}')

    # CCompute MDS
    mds = MDS(similarity_matrix)
    eigenvalues, eigenvectors = mds.eigenvalues, mds.eigenvectors
    print(f'Eigenvalues: {eigenvalues}')

    fig = mds.scree_plot()
    plt.show()
