import unittest
import numpy as np
from ..pca import PCA

class TestPCA(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[1, 3, 7, 2, -2, -2], [6, 8, -1, 4, 1, 7], [5, 9, 0, 3, 8, 6], [2, 0, 3, -3, 4, 8]])
        self.pca = PCA(self.data)
        self.pseudo_pca = PCA(self.data.T)
        self.p = self.data.shape[0]
        self.n = self.data.shape[1]


    def test_variance_eigenvalues(self):
        # Normal pca
        cov_matrix = np.cov(self.data)
        total_variance = np.trace(cov_matrix)
        self.assertAlmostEqual(np.sum(self.pca.eigenvalues), total_variance, places=5)

        # PCA with pseudo-covariance matrix
        cov_matrix = np.cov(self.data.T)
        total_variance = np.trace(cov_matrix)
        self.assertAlmostEqual(np.sum(self.pseudo_pca.eigenvalues), total_variance, places=5)


    def test_shape_eigenvalues(self):
        self.assertEqual(self.pca.eigenvalues.shape, (self.p,))
        self.assertEqual(self.pseudo_pca.eigenvalues.shape, (self.p,))


    def test_normalization_eigenvectors(self):
        for i in range(self.pca.eigenvectors.shape[1]):
            v = self.pca.eigenvectors[:, i]
            v2 = self.pseudo_pca.eigenvectors[:, i]
            self.assertAlmostEqual(np.linalg.norm(v), 1, places=5)
            self.assertAlmostEqual(np.linalg.norm(v2), 1, places=5)


    def test_shape_eigenvectors(self):
        self.assertEqual(self.pca.eigenvectors.shape, (self.p, self.p))
        self.assertEqual(self.pseudo_pca.eigenvectors.shape, (self.n, self.p))


    def test_shape_mean(self):
        self.assertEqual(self.pca.mean.shape, (self.p,))
        self.assertEqual(self.pseudo_pca.mean.shape, (self.n,))


    def test_eigendecomposition(self):
        # Normal pca
        for i in range(self.pca.eigenvalues.shape[0]):
            x_v = np.cov(self.data) @ self.pca.eigenvectors[:, i]
            lamb = x_v / self.pca.eigenvectors[:, i]
            good = np.allclose(lamb, self.pca.eigenvalues[i])
            self.assertTrue(good, 'Diagonalization with covariance matrix is erroneous.')

        # PCA with pseudo-covariance matrix
        for i in range(self.p - 1):
            x_v = np.cov(self.data.T) @ self.pseudo_pca.eigenvectors[:, i]
            lamb = x_v / self.pseudo_pca.eigenvectors[:, i]
            good = np.allclose(lamb, self.pseudo_pca.eigenvalues[i])
            self.assertTrue(good, 'Diagonalization with pseudo covariance matrix is erroneous.')


    def test_pca_transformations(self):
        # Normal pca
        transformed_data = self.pca.to_pca_space(self.data)
        reconstructed_data = self.pca.from_pca_space(transformed_data)
        reconstruction_error = np.linalg.norm(self.data - reconstructed_data)
        self.assertAlmostEqual(reconstruction_error, 0, places=5)


if __name__ == '__main__':
    unittest.main()