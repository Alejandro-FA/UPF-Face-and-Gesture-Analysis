import unittest
import numpy as np
from ..pca import PCA

class TestPCA(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[2, 9, 6, 8], [12, 9, 9, 10]])
        self.pca = PCA(self.data)
        self.p = self.data.shape[0]

    def test_eigenvalues(self):
        cov_matrix = np.cov(self.data)
        total_variance = np.trace(cov_matrix)
        self.assertAlmostEqual(np.sum(self.pca.eigenvalues), total_variance, places=5)

    def test_eigenvectors(self):
        for i in range(self.pca.eigenvectors.shape[1]):
            v = self.pca.eigenvectors[:, i]
            self.assertAlmostEqual(np.linalg.norm(v), 1, places=5)

    def test_mean(self):
        self.assertEqual(self.pca.mean.shape, (self.p,))

    def test_pca_transformations(self):
        transformed_data = self.pca.to_pca_space(self.data)
        reconstructed_data = self.pca.from_pca_space(transformed_data)
        reconstruction_error = np.linalg.norm(self.data - reconstructed_data)
        self.assertAlmostEqual(reconstruction_error, 0, places=5)


class TestPseudoPCA(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[2, 9], [6, 8], [12, 9], [9, 10]])
        self.pca = PCA(self.data)
        self.p = self.data.shape[0]

    def test_pca_transformations(self):
        transformed_data = self.pca.to_pca_space(self.data)
        reconstructed_data = self.pca.from_pca_space(transformed_data)
        reconstruction_error = np.linalg.norm(self.data - reconstructed_data)
        self.assertLess(reconstruction_error, 5)


if __name__ == '__main__':
    unittest.main()