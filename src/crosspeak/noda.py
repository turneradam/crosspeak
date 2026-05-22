import numpy as np

from crosspeak.series import SpectralSeries


def hilbert_noda_matrix(n):
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 2:
        raise ValueError(f"need at least 2 perturbation points, got {n}")

    indices = np.arange(n)
    diff = indices[None, :] - indices[:, None]

    matrix = np.zeros((n, n))
    off_diagonal = diff != 0
    matrix[off_diagonal] = 1.0 / (np.pi * diff[off_diagonal])

    return matrix


def synchronous(series):
    if not isinstance(series, SpectralSeries):
        raise TypeError(f"expected SpectralSeries, got {type(series).__name__}")

    m = series.n_perturbations
    Y = series.intensities - series.intensities.mean(axis=0)
    return (Y.T @ Y) / (m - 1)


def asynchronous(series):
    if not isinstance(series, SpectralSeries):
        raise TypeError(f"expected SpectralSeries, got {type(series).__name__}")

    m = series.n_perturbations
    Y = series.intensities - series.intensities.mean(axis=0)
    N = hilbert_noda_matrix(m)
    return (Y.T @ N @ Y) / (m - 1)
