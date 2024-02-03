import numpy as np
import matplotlib.pyplot as plt

#TODO: compute dissimilarity matrix
def dissimilarity_matrix(sim_matrix: np.ndarray):
    dissimilar_matrix = np.zeros(sim_matrix.shape)
    
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            dissimilar_matrix[i, j] = np.sqrt(sim_matrix[i, i] - 2 * sim_matrix[i, j] + sim_matrix[j, j])
    
    return dissimilar_matrix

    
def cart2pol(data: np.ndarray):
    x = data[0, :]
    y = data[1, :]
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    
    return np.array([r, phi])


def visualize_similarity_matrix(sim_matrix: np.ndarray, labels: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=(10, 7), num="Similarity matrix")
    
    plt.imshow(sim_matrix, cmap="rainbow")
    
    for i in range(len(labels)):
        plt.axhline(y = i * 3 - 0.5, color="black", linewidth=1.5)
        plt.axvline(x = i * 3 - 0.5, color="black", linewidth=1.5)
        
    plt.title("Similarity matrix visualization", fontsize=14, fontweight="bold")
    ticks_positions = np.arange(1, 23, 3)
    
    plt.gca().set_xticks(ticks_positions, labels=labels)
    plt.gca().set_yticks(ticks_positions, labels=labels)
    plt.colorbar()
    plt.xticks(rotation=45)
    
    
    
    return fig