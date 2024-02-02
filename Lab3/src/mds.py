import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import f


class MDS:
    def __init__(self, distances: np.ndarray) -> None:
        assert distances.ndim == 2, "The input should be a 2D array containing the distances between the samples"
        assert distances.shape[0] == distances.shape[1], "The input matrix should be square"
        self.__distances = distances
        B = MDS.__get_B(distances)
        self.__eigenvalues, self.__eigenvectors = MDS.__eig(B)
    

    def to_mds_space(self, n: int=None) -> np.ndarray:
        # TODO: Return principal coordinates in MDS space
        n = n if n else np.sum(self.__eigenvalues > 0)
        y = np.zeros((self.__eigenvectors.shape[0], n))
        for i in range(n):
            y[:, i] = self.__eigenvectors[:, i] * np.sqrt(self.__eigenvalues[i])
        
        return y.T


    def from_mds_space(self) -> np.ndarray:
        # TODO: Return the distances in the mds space
        pass

    
    def __confidence_region(self, alpha: float=0.05) -> Ellipse:
        # # Define the parameters for the F distribution
        # p = 2  # number of dimensions
        # n = self.__distances.shape[1]  # number of samples

        # # Get the inverse of the cumulative distribution function of F
        # x = f.ppf(alpha, p, n-p)
        # c = ((n - 1) * p / (n - p)) * x

        # # Compute the confidence region
        # semi_major = np.sqrt(self.__eigenvalues[0] / n) * np.sqrt(c)
        # semi_minor = np.sqrt(self.__eigenvalues[1] / n) * np.sqrt(c)
        # Compute the confidence region
        semi_major = np.sqrt(self.__eigenvalues[0])
        semi_minor = np.sqrt(self.__eigenvalues[1])
        center = (0.0, 0.0)
        return Ellipse(center, 2 * semi_major, 2 * semi_minor, edgecolor='r', fc='None', lw=2)
    

    def space_plot(self, colors: list[str], emotions: list[str], flip_x: bool=False, flip_y: bool=False) -> plt.Figure:
        assert len(colors) == len(emotions), "The number of colors should be the same as the number of emotions"
        mds_coordinates = self.to_mds_space(n=2)
        x_coords = mds_coordinates[0, :] * -1 if flip_x else mds_coordinates[0, :]
        y_coords = mds_coordinates[1, :] * -1 if flip_y else mds_coordinates[1, :]
        
        # Create a figure and plot the points in MDS coordinates
        fig = plt.figure(figsize=(10, 5), num="MDS space")
        plt.title("Emotions in the MDS space")
        group_every = self.eigenvalues.shape[0] // len(colors)
        for i in np.arange(0, self.eigenvalues.shape[0], group_every):
            plt.scatter(
                x_coords[i:i+group_every],
                y_coords[i:i+group_every],
                color=colors[i // group_every] if colors is not None else None,
                label=emotions[i // group_every] if emotions is not None else None
            )
        
        # Plot the confidence ellipse
        ellipse = self.__confidence_region()        
        
        plt.gca().add_patch(ellipse)
        plt.gca().set_aspect("equal")
        plt.grid()
        plt.xlabel("Valence")
        plt.ylabel("Arousal")
        if emotions is not None:
            plt.legend()
        
        return fig
        

    def scree_plot(self, max_eigenvalues: int=None, num_resamples: int=0) -> plt.Figure:
        #TODO: perhaps avoid plotting eigenvalues that are 0
        max_eigenvalues = max_eigenvalues if max_eigenvalues else self.__eigenvalues.shape[0]
        original_total_variance = np.sum(self.__eigenvalues)
        x = np.arange(1, len(self.__eigenvalues[0:max_eigenvalues]) + 1, dtype=int)
        y = self.__eigenvalues[0:max_eigenvalues] / original_total_variance

        fig = plt.figure(figsize=(10, 5), num="Scree plot")
        
        plt.xlabel("Eigenvalue index")
        plt.ylabel("$\lambda_i$ (ratio of total variance)")
        ax = plt.gca()  # get the current axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        title = "Scree plot"

        if num_resamples > 0:
            bootstrap_eigs = _bootstrap_mds(self.__distances, b=num_resamples)
            assert np.all(np.isreal(bootstrap_eigs)), "All bootstrap eigs should be real"
            mean_bootstrap_eigs = np.mean(bootstrap_eigs, axis=1)
            for i in range(bootstrap_eigs.shape[1]):
                total_variance = np.sum(bootstrap_eigs[:, i])
                y2 = bootstrap_eigs[0:max_eigenvalues, i] / total_variance
                plt.plot(x, y2, color="red", linewidth=0.25, alpha=0.4, label=f"Bootstrap eigenvalues (b={num_resamples})" if i == 0 else None)

            alpha = 0.1 # We use significance level of 0.1 because the number of samples is very low
            num_significant = self.get_significant_eigenvalues(bootstrap_eigs, alpha)
            title += f', (significant components (alpha {alpha}): {num_significant})'
            plt.plot(x, mean_bootstrap_eigs[0:max_eigenvalues] / np.sum(mean_bootstrap_eigs), linewidth=1 , color="green", label="Mean bootstap eigenvalues")
        
        plt.plot(x, y, marker="*", linewidth=1.25, markersize=3, color="blue", label="Original eigenvalues")

        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid()
        
        return fig
    
    

    def get_significant_eigenvalues(self, bootstrap_eig: list[np.ndarray], alpha: float=0.05) -> int:
        p = self.eigenvalues.shape[0]
        b = len(bootstrap_eig)
        assert p == bootstrap_eig.shape[0], "The number of eigenvalues should be the same"

        # Compute how many bootstrap eigenvalues are greater than the original ones (by chance)
        original_eigs = self.eigenvalues / np.sum(self.eigenvalues)
        eigs = bootstrap_eig / np.sum(bootstrap_eig, axis=0)
        
        significant = 0
        for i in range(p):
            count = np.sum(eigs[i, :] > original_eigs[i])
            p_value = (count + 1) / (b + 1)
            if p_value > alpha:
                break
            significant += 1
        
        return significant
    

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
        # IMPORTANT NOTE: Negative eigenvalues are set to 0 instead of being removed.
        # The respective eigenvectors are also set to a vector of 0s.
        eigenvalues, eigenvectors = np.linalg.eig(x)
        eigenvalues, eigenvectors = MDS.__clean_eigenvalues(eigenvalues, eigenvectors)

        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors
    

    @staticmethod
    def __get_B(distances: np.ndarray) -> np.ndarray:
        n = distances.shape[0]
        H = np.eye(n) - (1/n) * np.ones((n, n))
        A = -0.5 * np.square(distances)
        B = H @ A @ H
        assert np.allclose( np.mean(B, axis=0), 0), "The mean of the rows of B should be 0"
        assert np.allclose( np.mean(B, axis=1), 0), "The mean of the columns of B should be 0"
        assert np.allclose(B, B.T), "Matrix B should be symmetric"
        return B
    

    @staticmethod
    def __clean_eigenvalues(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Eigendecomposing matrix B can introduce some numerical errors, yielding complex eigenvalues which do not make sense as
        # matrix B is symmetric. Thus, we discard complex eigenvalues.
        mask = np.iscomplex(eigenvalues)
        eigenvalues[mask] = 0
        eigenvectors[:, mask] = 0
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Negative eigenvalues are also discarded as they do not make sense in the context of MDS
        mask = eigenvalues < 0
        eigenvalues[mask] = 0
        eigenvectors[:, mask] = 0

        assert np.all(np.isreal(eigenvalues)), "All eigenvalues should be real numbers"
        assert np.all(np.isreal(eigenvectors)), "All eigenvectors should be real numbers"
        assert np.all(eigenvalues >= 0), "All eigenvalues should be non-negative"
        return eigenvalues, eigenvectors
    

def _bootstrap_mds(distances: np.ndarray, b: int=1000) -> list[np.ndarray]:
    all_eigenvalues = [None] * b
    row_idx, col_idx = np.triu_indices(distances.shape[0], 1)
  
    # Generate bootstrap samples under the hypothesis that the distances
    # between the samples are independent and identically distributed
    for i in range(b):
        new_i = np.random.choice(row_idx, len(row_idx))
        new_j = np.random.choice(col_idx, len(col_idx))
        bootstrap_sample = np.zeros(distances.shape) # Distance of one image to itself is always considered to be 0
        for row in range(0, distances.shape[0] - 1):
            for col in range(row + 1, distances.shape[1]):
                bootstrap_sample[row, col] = distances[new_i[row], new_j[col]]
                bootstrap_sample[col, row] = bootstrap_sample[row, col]

        bootstrap_mds = MDS(bootstrap_sample)
        all_eigenvalues[i] = bootstrap_mds.eigenvalues
    
    return np.stack(all_eigenvalues, axis=1)

