import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class PCA:
    """
    Principal Component Analysis (PCA) class for dimensionality reduction and feature extraction.

    Args:
        data (np.ndarray): The input data matrix. Expects the data to be a 2D matrix, where the number of rows
            represents the number of dimensions of the data and the number of columns corresponds to the number of samples.

    Attributes:
        eigenvalues (np.ndarray): The eigenvalues of the covariance or pseudocovariance matrix.
        eigenvectors (np.ndarray): The eigenvectors of the covariance or pseudocovariance matrix.
        mean (np.ndarray): The mean vector of the input data.
        data (np.ndarray): The input data matrix.

    Methods:
        to_pca_space(x, num_components=None): Projects the input data into the PCA space.
        from_pca_space(x): Projects the data from the PCA space back to the original space.
        scree_plot(): Generates a scree plot to visualize the explained variance of each principal component.
    """

    def __init__(self, data: np.ndarray) -> None:
        """
        Initializes the PCA object.

        Args:
            data (np.ndarray): The input data matrix.
        """
        use_pseudocovariance = data.shape[0] > data.shape[1]
        self.__data = data
        self.__mean = PCA.__get_mean(data)
        self.__eigenvalues, self.__eigenvectors = PCA.__eig(data, use_pseudocovariance)
    

    def to_pca_space(self, x: np.ndarray, num_components: int=None) -> np.ndarray:
        """
        Projects the input data into the PCA space.

        Args:
            x (np.ndarray): The input data matrix to be projected.
            num_components (int, optional): The number of principal components to keep. If not provided,
                all the principal components will be used.

        Returns:
            np.ndarray: The projected data in the PCA space.
        """
        p = num_components if num_components else x.shape[0]
        return self.__eigenvectors[:, :p].T @ (x - self.__mean)


    def from_pca_space(self, x: np.ndarray) -> np.ndarray:
        """
        Projects the data from the PCA space back to the original space.

        Args:
            x (np.ndarray): The data in the PCA space to be projected back.

        Returns:
            np.ndarray: The projected data in the original space.
        """
        p = x.shape[0]
        assert p <= self.__eigenvectors.shape[1], 'The number of rows in the input data must be at most the number of columns in the eigenvector matrix'
        return (self.__eigenvectors[:, 0:p] @ x) + self.__mean
    

    def scree_plot(self, max_eigenvalues: int=None, num_permutations: int=0) -> plt.Figure:
        """
        Generates a scree plot to visualize the explained variance of each principal component.

        Returns:
            plt.Figure: The scree plot figure.
        """
        max_eigenvalues = max_eigenvalues if max_eigenvalues else len(self.__eigenvalues)
        total_variance = np.sum(self.__eigenvalues)
        x = np.arange(1, len(self.__eigenvalues[0:max_eigenvalues]) + 1, dtype=int)
        y = self.__eigenvalues[0:max_eigenvalues] / total_variance

        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker="*", linewidth=1.25, markersize=3, color="blue", label="Original eigenvalues")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("$\lambda_i$ (ratio of total variance))")
        title = "Scree plot"

        if num_permutations > 0:
            bootstrap_eigs = _bootstrap_pca(self.data, b=num_permutations)
            for i in range(bootstrap_eigs.shape[1]):
                y2 = bootstrap_eigs[0:max_eigenvalues, i] / total_variance
                plt.plot(x, y2, color="red", linewidth=0.75, alpha=0.4, label=f"Bootstrap eigenvalues (b={num_permutations})" if i == 0 else None)

            alpha = 0.05
            num_significant = self.get_significant_eigenvalues(bootstrap_eigs, alpha)
            title += f', (significant components (alpha {alpha}): {num_significant})'

        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid()
        
        return fig
    

    def get_significant_eigenvalues(self, bootstrap_eig: np.ndarray, alpha: float=0.05) -> int:
        # Expects bootstrap_eig to be a 2D matrix, where the number of rows represents the number of dimensions of the data and the number of columns corresponds to the number of bootstrap samples.
        p = self.eigenvalues.shape[0]
        b = len(bootstrap_eig)
        p_values = np.zeros(p)

        # Compute how many bootstrap eigenvalues are greater than the original ones (by chance)
        for i in range(p):
            count = np.sum(bootstrap_eig[i, :] > self.eigenvalues[i])
            p_values[i] = (count + 1) / (b + 1)
        
        return np.sum(p_values < alpha)


    @property
    def eigenvalues(self) -> np.ndarray:
        """
        np.ndarray: The eigenvalues of the covariance or pseudocovariance matrix.
        """
        return self.__eigenvalues
    

    @property
    def eigenvectors(self) -> np.ndarray:
        """
        np.ndarray: The eigenvectors of the covariance or pseudocovariance matrix.
        """
        return self.__eigenvectors
    

    @property
    def mean(self) -> np.ndarray:
        """
        np.ndarray: The mean vector of the input data.
        """
        return self.__mean.flatten()
    
    
    @property
    def data(self) -> np.ndarray:
        """
        np.ndarray: The input data matrix.
        """
        return self.__data
    

    @staticmethod
    def __eig(x: np.ndarray, use_pseudocovariance: bool=False):
        """
        Computes the eigenvalues and eigenvectors of the covariance or pseudocovariance matrix.

        Args:
            x (np.ndarray): The input data matrix.
            use_pseudocovariance (bool, optional): Determines whether to compute the eigendecomposition of the
                covariance matrix (False) or the pseudocovariance matrix (True). Default is False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The eigenvalues and eigenvectors (sorted).
        """
        cov_matrix = PCA.__get_cov_matrix(x, use_pseudocovariance)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if use_pseudocovariance:
            eigenvectors = x @ eigenvectors
            eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
        
        return eigenvalues, eigenvectors
    

    @staticmethod
    def __get_cov_matrix(x: np.ndarray, use_pseudocovariance: bool=False) -> np.ndarray:
        """
        Computes the covariance or pseudocovariance matrix.

        Args:
            x (np.ndarray): The input data matrix.
            use_pseudocovariance (bool, optional): Determines whether to compute the covariance matrix (False)
                or the pseudocovariance matrix (True). Default is False.

        Returns:
            np.ndarray: The covariance or pseudocovariance matrix.
        """
        p = x.shape[0]
        n = x.shape[1]
        mean = PCA.__get_mean(x)

        if use_pseudocovariance:
            return 1/(p-1) * (x - mean).T @ (x - mean)
        else:
            return 1/(n-1) * (x - mean) @ (x - mean).T


    @staticmethod
    def __get_mean(x: np.ndarray) -> np.ndarray:
        """
        Computes the mean vector of the input data.

        Args:
            x (np.ndarray): The input data matrix.

        Returns:
            np.ndarray: The mean vector.
        """
        p = x.shape[0]
        return np.mean(x, axis=1).reshape(p, 1)

        

def _bootstrap_pca(data: np.ndarray, b: int=100) -> np.ndarray:
    rng = np.random.default_rng()
    all_eigenvalues = [None] * b
    desc = 'Computing PCA of permuted samples'

    for i in tqdm(range(b), desc=desc):
        permuted = rng.permuted(data, axis=0)
        bootstrap_pca = PCA(permuted)
        all_eigenvalues[i] = bootstrap_pca.eigenvalues
    
    return np.stack(all_eigenvalues, axis=1)