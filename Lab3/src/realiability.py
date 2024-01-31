import numpy as np


def reliability(similarity_mat: np.ndarray, consistency_mat: np.ndarray, b: int=1000) -> tuple[float, float]:
    # Get only the relevant values to check the reliability
    similarity_mat = __upper_diagonal(similarity_mat).flatten()
    consistency_mat = __upper_diagonal(consistency_mat).flatten()
    
    # Get the values of the similarity matrix that correspond to the non-NaN values of the consistency matrix
    non_nan_idx = __get_non_nan_idx(consistency_mat)
    similarity_values = similarity_mat[non_nan_idx]
    consistency_values = consistency_mat[non_nan_idx]

    # Compute the error of the original consistency matrix
    error = __mean_absolute_error(similarity_values, consistency_values)
    
    # Perform bootstrap and compute the errors of the bootstrap samples
    bootstap_errors = [None] * b
    for i in range(b):
        bootstrap_sample = np.random.choice(consistency_values, size=len(consistency_values), replace=True)
        bootstap_errors[i] = __mean_absolute_error(similarity_values, bootstrap_sample)

    # Compute the p-value
    count = np.sum(bootstap_errors < error)
    p_value = (count + 1) / (b + 1)
    return error, p_value


def __get_non_nan_idx(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 1, "The input array must be 1-dimensional"
    return np.where(np.isfinite(x))


def __mean_absolute_error(x: np.ndarray, y: np.ndarray) -> float:
    return np.mean(np.abs(x - y))


def __upper_diagonal(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, "Input should be a 2D array"
    i, j = np.triu_indices(x.shape[0], 1)
    return x[i, j]