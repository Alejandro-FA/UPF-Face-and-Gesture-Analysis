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
    

    def to_mds_space(self, n: int) -> np.ndarray:
        # TODO: Return principal coordinates in MDS space
        pass


    def from_mds_space(self) -> np.ndarray:
        # TODO: Return the distances in the mds space
        pass


    def scree_plot(self, max_eigenvalues: int=None, num_resamples: int=0) -> plt.Figure:
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
            for i in range(len(bootstrap_eigs)):
                max_eigenvalues = len(bootstrap_eigs[i])
                x = np.arange(1, len(bootstrap_eigs[i][0:max_eigenvalues]) + 1, dtype=int)
                
                total_variance = np.sum(bootstrap_eigs[i])
                y2 = bootstrap_eigs[i][0:max_eigenvalues] / total_variance
                plt.plot(x, y2, color="red", linewidth=0.75, alpha=0.4, label=f"Bootstrap eigenvalues (b={num_resamples})" if i == 0 else None)

            alpha = 0.05
            num_significant = self.get_significant_eigenvalues(bootstrap_eigs, alpha)
            title += f', (significant components (alpha {alpha}): {num_significant})'

        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid()
        
        return fig
    
    

    def get_significant_eigenvalues(self, bootstrap_eig: list[np.ndarray], alpha: float=0.05) -> int:
        p = self.eigenvalues.shape[0]
        b = len(bootstrap_eig)
        p_values = np.zeros(p)

        # Compute how many bootstrap eigenvalues are greater than the original ones (by chance)
        original_eigs = self.eigenvalues / np.sum(self.eigenvalues)
        # eigs = bootstrap_eig / np.sum(bootstrap_eig, axis=0)
        print(len(original_eigs))
        eigs = [[]] * b
        for i in range(b):
            total_sum = np.sum(bootstrap_eig[i])
            eigs[i] = bootstrap_eig[i] / total_sum
        
        print([len(eig) for eig in eigs])
        for i in range(p):
            count = 0
            for j in range(b):
                for k in range(len(eigs[j])):
                    if eigs[j][k] > original_eigs[i]:
                        count += 1
                    
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
        
        # return eigenvalues, eigenvectors
        
        #TODO: remove negative eigenvalues (if any)
        non_negative_eigenvalues = []
        non_negative_eigenvectors = []
        for i, eig in enumerate(eigenvalues):
            if eig > 0:
                non_negative_eigenvalues.append(eig)
                non_negative_eigenvectors.append(eigenvectors[:, i])
                
        return np.asarray(non_negative_eigenvalues), np.asarray(non_negative_eigenvectors)
    

    @staticmethod
    def __get_B(distances: np.ndarray) -> np.ndarray:
        n = distances.shape[0]
        H = np.eye(n) - (1/n) * np.ones((n, n))
        A = -0.5 * np.square(distances)
        B = H @ A @ H
        assert np.allclose( np.mean(B, axis=0), 0), "The mean of the rows of B should be 0"
        assert np.allclose( np.mean(B, axis=1), 0), "The mean of the columns of B should be 0"
        return B
    

def _bootstrap_mds(distances: np.ndarray, b: int=1000) -> list[np.ndarray]:
    all_eigenvalues = [None] * b
    original_shape = distances.shape
    flat_distances = distances.flatten()
    
    row_idx, col_idx = np.triu_indices(distances.shape[0], 1)
  
    for i in range(b):
        # FIXME: This yields to complex eigenvalues. I believe that the correct resample has to be built as follows:
        #       - permute the upper triangular part
        #       - make the matrix symmetric by copying the data from the upper triangle to the lower triangle
        # https://search.r-project.org/CRAN/refmans/MultBiplotR/html/BootstrapSmacof.html
        # new_i = np.random.choice(row_idx, len(row_idx))
        # new_j = np.random.choice(col_idx, len(col_idx))
        # bootstrap_sample = np.zeros(distances.shape)
        # for row in range(distances.shape[0] - 1):
        #     for col in range(row, distances.shape[1]):
        #         bootstrap_sample[row, col] = distances[new_i[row], new_j[col]]
        #         bootstrap_sample[col, row] = bootstrap_sample[row, col]
                
        # assert np.array_equal(bootstrap_sample, bootstrap_sample.T), "Bootstrap resample is not symmetric"
        bootstrap_sample = np.random.choice(flat_distances, size=distances.shape, replace=True)
        bootstrap_sample = bootstrap_sample.reshape(original_shape)
        bootstrap_mds = MDS(bootstrap_sample)
        all_eigenvalues[i] = np.real(bootstrap_mds.eigenvalues)
        
    # np.stack cannot be used as each bootstrap resample may not have the same number of eigenvalues (as we are removing the negative ones)
    return all_eigenvalues

