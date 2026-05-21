import numpy as np


def hilbert_noda_matrix(n):
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 2:
        raise ValueError(f"need at least 2 perturbation points, got {n}")

    indices = np.arange(n)
    # diff[j, k] = k - j
    diff = indices[None, :] - indices[:, None]

    matrix = np.zeros((n, n))
    off_diagonal = diff != 0
    matrix[off_diagonal] = 1.0 / (np.pi * diff[off_diagonal])

    return matrix
