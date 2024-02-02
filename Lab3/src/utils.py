import numpy as np

#TODO: compute dissimilarity matrix
def dissimilarity_matrix(sim_matrix: np.ndarray):
    dissimilar_matrix = np.zeros(sim_matrix.shape)
    
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            dissimilar_matrix[i, j] = np.sqrt(sim_matrix[i, i] - 2 * sim_matrix[i, j] + sim_matrix[j, j])
    
    return dissimilar_matrix

    
    