import numpy as np
import matplotlib.pyplot as plt


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

def similarity_matrix_sort_emotions(similarity_matrix: np.ndarray, emotions: list[str], emotions_sorted: list[str]) -> np.ndarray:
    sorted_sim_matrix = np.zeros(similarity_matrix.shape)
    for i, new_row in enumerate(np.arange(0, 24, 3)):
        for j, new_col in enumerate(np.arange(0, 24, 3)):
            idx_1 = emotions.index(emotions_sorted[i]) * 3
            idx_2 = emotions.index(emotions_sorted[j]) * 3
            
            sorted_sim_matrix[new_row:new_row+3, new_col:new_col+3] = similarity_matrix[idx_1:idx_1 + 3, idx_2:idx_2 + 3]
            
    return sorted_sim_matrix
    


def visualize_similarity_matrix(sim_matrix: np.ndarray, labels: list[str], title: str ="Similarity matrix visualization") -> plt.Figure:
    fig = plt.figure(figsize=(10, 7), num=title)
    
    plt.imshow(sim_matrix, cmap="rainbow")
    
    for i in range(len(labels)):
        plt.axhline(y = i * 3 - 0.5, color="black", linewidth=1.5)
        plt.axvline(x = i * 3 - 0.5, color="black", linewidth=1.5)
        
    plt.title(title, fontsize=14, fontweight="bold")
    ticks_positions = np.arange(1, 23, 3)
    
    plt.gca().set_xticks(ticks_positions, labels=labels)
    plt.gca().set_yticks(ticks_positions, labels=labels)
    plt.colorbar()
    plt.xticks(rotation=45)
    
    
    
    return fig