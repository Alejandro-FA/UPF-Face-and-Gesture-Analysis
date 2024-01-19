import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, x: np.ndarray) -> None:
        # Expects the data to be a 2D matrix, where the number of rows
        # represents the number of dimensions of the data and the number of
        # columns corresponds to the number of samples
        self.__mean = PCA.__get_mean(x)
        use_pseudocovariance = x.shape[0] > x.shape[1]
        self.__eigenvalues, self.__eigenvectors = PCA.__eig(x, use_pseudocovariance)
    

    def to_pca_space(self, x: np.ndarray, num_components: int=None) -> np.ndarray:
        p = num_components if num_components else x.shape[0]
        return self.__eigenvectors[:, 0:p].T @ (x - self.__mean)


    def from_pca_space(self, x: np.ndarray, num_components: int=None) -> np.ndarray:
        p = num_components if num_components else x.shape[0]
        return (self.__eigenvectors[:, 0:p] @ x) + self.__mean
    

    def scree_plot(self) -> plt.Figure:
        total_variance = np.sum(self.__eigenvalues)
        x = np.arange(1, len(self.__eigenvalues) + 1)
        y = self.__eigenvalues / total_variance

        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker="*", linewidth=0.5)
        plt.xlabel("i")
        plt.ylabel("$\lambda_i$ (percentage of total variance))")
        plt.title("Scree plot", fontsize=14, fontweight="bold")
        plt.grid()
        return fig


    @property
    def eigenvalues(self) -> np.ndarray:
        return self.__eigenvalues
    

    @property
    def eigenvectors(self) -> np.ndarray:
        return self.__eigenvectors
    

    @staticmethod
    def __eig(x: np.ndarray, use_pseudocovariance: bool=False):
        # Return eigenvalues and eigenvectors
        # Determine whether to eigendecompose the covariance or pseudocovariance matrix
        cov_matrix = PCA.__get_cov_matrix(x, use_pseudocovariance)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if use_pseudocovariance:
            eigenvectors = x @ eigenvectors
        
        return eigenvalues, eigenvectors
    

    @staticmethod
    def __get_cov_matrix(x: np.ndarray, use_pseudocovariance: bool=False) -> np.ndarray:
        p = x.shape[0]
        n = x.shape[1]
        mean = PCA.__get_mean(x)

        if use_pseudocovariance:
            return 1/(p-1) * (x - mean).T @ (x - mean)
        else:
            return 1/(n-1) * (x - mean) @ (x - mean).T


    @staticmethod
    def __get_mean(x: np.ndarray) -> np.ndarray:
        p = x.shape[0]
        return np.mean(x, axis=1).reshape(p, 1)
