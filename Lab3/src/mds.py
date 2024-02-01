import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class MDS:
    def __init__(self, distances: np.ndarray) -> None:
        assert distances.ndim == 2, "The input should be a 2D array containing the distances between the samples"
        assert distances.shape[0] == distances.shape[1], "The input matrix should be square"
        self.__distances = distances
        B = MDS.__get_B(distances)
        self.__eigenvalues, self.__eigenvectors = MDS.__eig(B)
        self.significant_components = 0
    

    def to_mds_space(self, n: int) -> np.ndarray:
        # TODO: Return principal coordinates in MDS space
        y = np.zeros((self.__eigenvectors.shape[0], n))
        print(y.shape)
        for i in range(n):
            print(self.__eigenvalues[i])
            y[:, i] = self.__eigenvectors[:, n] * np.sqrt(self.__eigenvalues[i])
        
        return y.T


    def from_mds_space(self) -> np.ndarray:
        # TODO: Return the distances in the mds space
        pass


    def scree_plot(self, max_eigenvalues: int=None, num_resamples: int=0) -> plt.Figure:
        max_eigenvalues = max_eigenvalues if max_eigenvalues else len(self.__eigenvalues)
        original_total_variance = np.sum(self.__eigenvalues)
        x = np.arange(1, len(self.__eigenvalues[0:max_eigenvalues]) + 1, dtype=int)
        y = self.__eigenvalues[0:max_eigenvalues] / original_total_variance

        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker="*", linewidth=1.25, markersize=3, color="blue", label="Original eigenvalues")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("$\lambda_i$ (ratio of total variance)")
        ax = plt.gca()  # get the current axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        title = "Scree plot"

        if num_resamples > 0:
            bootstrap_eigs = _bootstrap_mds(self.__distances, b=num_resamples)
            for i in range(bootstrap_eigs.shape[1]):
                total_variance = np.sum(bootstrap_eigs[:, i])
                y2 = bootstrap_eigs[0:max_eigenvalues, i] / total_variance
                plt.plot(x, y2, color="red", linewidth=0.75, alpha=0.4, label=f"Bootstrap eigenvalues (b={num_resamples})" if i == 0 else None)

            alpha = 0.05
            num_significant = self.get_significant_eigenvalues(bootstrap_eigs, alpha)
            self.significant_components = num_significant
            title += f', (significant components (alpha {alpha}): {num_significant})'

        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid()
        
        return fig
    
    

    def get_significant_eigenvalues(self, bootstrap_eig: np.ndarray, alpha: float=0.05) -> int:
        p = self.eigenvalues.shape[0]
        b = len(bootstrap_eig)
        p_values = np.zeros(p)

        # Compute how many bootstrap eigenvalues are greater than the original ones (by chance)
        original_eigs = self.eigenvalues / np.sum(self.eigenvalues)
        eigs = bootstrap_eig / np.sum(bootstrap_eig, axis=0)
        for i in range(p):
            count = np.sum(eigs[i, :] > original_eigs[i])
            p_values[i] = (count + 1) / (b + 1)
        
        return np.sum(p_values < alpha)
    

    # TODO: do distance distance plot
    def distance_distance_plot(self) -> plt.Figure:
        pass


    @property
    def eigenvalues(self) -> np.ndarray:
        return self.__eigenvalues
    

    @property
    def eigenvectors(self) -> np.ndarray:
        return self.__eigenvectors
    
    
    @property
    def distances(self) -> np.ndarray:
        return self.__distances
    

    @staticmethod
    def __eig(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues, eigenvectors = np.linalg.eig(x)

        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        #TODO: remove negative eigenvalues (if any)
        return eigenvalues, eigenvectors
    

    @staticmethod
    def __get_B(distances: np.ndarray) -> np.ndarray:
        n = distances.shape[0]
        H = np.eye(n) - (1/n) * np.ones((n, n))
        A = -0.5 * np.square(distances)
        B = H @ A @ H
        assert np.allclose( np.mean(B, axis=0), 0), "The mean of the rows of B should be 0"
        assert np.allclose( np.mean(B, axis=1), 0), "The mean of the columns of B should be 0"
        return B
    

def _bootstrap_mds(distances: np.ndarray, b: int=1000) -> np.ndarray:
    all_eigenvalues = [None] * b
    original_shape = distances.shape
    flat_distances = distances.flatten()
    
    for i in range(b):
        bootstrap_sample = np.random.choice(flat_distances, size=distances.shape, replace=True)
        bootstrap_sample = bootstrap_sample.reshape(original_shape)
        bootstrap_mds = MDS(bootstrap_sample)
        
        all_eigenvalues[i] = bootstrap_mds.eigenvalues
    
    return np.stack(all_eigenvalues, axis=1)