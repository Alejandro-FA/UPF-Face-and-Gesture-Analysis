import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, x: np.ndarray) -> None:
        # Expects the data to be a 2D matrix, where the number of rows
        # represents the number of dimensions of the data and the number of
        # columns corresponds to the number of samples
        self.__mean = PCA.__get_mean(x)
        self.__eigenvalues, self.__eigenvectors = PCA.__eig(x)

    def principal_components(self, x: np.ndarray, num_components) -> np.ndarray:
        return self.__eigenvectors[:, 0:num_components].T @ (x - self.__mean)
    
    def to_pca_space(self, x: np.ndarray) -> np.ndarray:
        return self.principal_components(x, x.shape[0])

    def from_pca_space(self, x: np.ndarray) -> np.ndarray:
        return (self.__eigenvalues @ x) + self.__mean
    
    def scree_plot(self):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.__eigenvalues)), self.__eigenvalues)
        plt.scatter(range(len(self.__eigenvalues)), self.__eigenvalues, marker="*")
        plt.xlabel("i")
        plt.ylabel("$\lambda_i$")
        plt.title("Scree plot", fontsize=14, fontweight="bold")
        plt.show()

    @property
    def eigenvalues(self) -> np.ndarray:
        return self.__eigenvalues
    
    @property
    def eigenvectors(self) -> np.ndarray:
        return self.__eigenvectors
    
    @staticmethod
    def __eig(x: np.ndarray):
        # Return eigenvalues and eigenvectors
        # Determine whether to eigendecompose the covariance or pseudocovariance matrix
        cov_matrix = PCA.__get_cov_matrix(x)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors
    
    @staticmethod
    def __get_cov_matrix(x: np.ndarray) -> np.ndarray:
        p = x.shape[0]
        n = x.shape[1]
        mean = PCA.__get_mean(x)

        if p <= n:
            # Use "normal" covariance matrix
            return 1/(n-1) * (x - mean) @ (x - mean).T
        else:
            # Use pseudo-covariance matrix
            return 1/(p-1) * (x - mean).T @ (x - mean)

    @staticmethod
    def __get_mean(x: np.ndarray) -> np.ndarray:
        p = x.shape[0]
        return np.mean(x, axis=1).reshape(p, 1)
