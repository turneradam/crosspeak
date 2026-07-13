from typing import NamedTuple

import numpy as np
from scipy.signal import find_peaks

from crosspeak.series import SpectralSeries


def hilbert_noda_matrix(n):
    """The Hilbert-Noda matrix for `n` perturbation points.

    The discrete kernel that turns the asynchronous calculation into a matrix
    product: with N in hand, Ψ = Y^T @ N @ Y / (m - 1). Off the diagonal
    N_jk = 1 / (π(k - j)); the diagonal is zero, since a point has no phase
    lag against itself.

    The sign convention matters. With N positive in the upper triangle, a
    positive Ψ(ν₁, ν₂) alongside a positive Φ means ν₁ leads ν₂ under Noda's
    rules. Transpose N and every ordering flips — so if this ever gets swapped
    out, check the sign against a sequence you already know.

    Parameters
    ----------
    n
        Number of perturbation points (m in the usual notation), at least 2.

    Returns
    -------
    np.ndarray
        The (n, n) matrix, antisymmetric with a zero diagonal.
    """
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
    """Synchronous 2DCOS matrix Φ of a spectral series.

    Φ(ν₁, ν₂) = Y^T @ Y / (m - 1) over the mean-centred intensities, which is
    just the sample covariance across the perturbation. Large positive values
    mean two wavenumbers rise and fall together; negative means they move in
    opposition. The diagonal is the autopower spectrum — where the perturbation
    does the most work.

    Parameters
    ----------
    series
        A `SpectralSeries` with at least two perturbation points.

    Returns
    -------
    np.ndarray
        The (n, n) synchronous matrix, symmetric by construction.
    """
    if not isinstance(series, SpectralSeries):
        raise TypeError(f"expected SpectralSeries, got {type(series).__name__}")

    m = series.n_perturbations
    Y = series.intensities - series.intensities.mean(axis=0)
    return (Y.T @ Y) / (m - 1)


def asynchronous(series):
    """Asynchronous 2DCOS matrix Ψ of a spectral series.

    Ψ(ν₁, ν₂) = Y^T @ N @ Y / (m - 1), with N the Hilbert-Noda matrix. Where Φ
    says which bands move together, Ψ says in what order: read against the sign
    of Φ, a nonzero Ψ resolves which of two overlapping features responds
    first. Antisymmetric, with a zero diagonal.

    Same mean-centring and conventions as `synchronous`; see
    `hilbert_noda_matrix` for the sign that fixes the direction of the ordering.

    Parameters
    ----------
    series
        A `SpectralSeries` with at least two perturbation points.

    Returns
    -------
    np.ndarray
        The (n, n) asynchronous matrix.
    """
    if not isinstance(series, SpectralSeries):
        raise TypeError(f"expected SpectralSeries, got {type(series).__name__}")

    m = series.n_perturbations
    Y = series.intensities - series.intensities.mean(axis=0)
    N = hilbert_noda_matrix(m)
    return (Y.T @ N @ Y) / (m - 1)


class AutopeakResult(NamedTuple):
    """Autopeaks found on a synchronous diagonal.

    Attributes
    ----------
    positions
        Wavenumbers at which autopeaks were detected, in the order of the
        input wavenumber axis.
    intensities
        Autopower values (diagonal of the synchronous matrix) at those
        positions.
    """

    positions: np.ndarray
    intensities: np.ndarray


def find_autopeaks(
    phi: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    prominence_frac: float = 0.02,
    **kwargs,
) -> AutopeakResult:
    """Detect autopeaks on the diagonal of a synchronous 2DCOS matrix.

    The diagonal of Phi is the autopower spectrum — the variance of each
    wavenumber across the perturbation series. Local maxima of this spectrum
    are Noda's autopeaks and identify the wavenumbers that change most
    strongly under the perturbation.

    Parameters
    ----------
    phi
        Synchronous 2DCOS matrix, shape (n, n), as returned by `synchronous()`.
    wavenumbers
        Wavenumber axis, shape (n,), strictly monotonic.
    prominence_frac
        Minimum peak prominence, as a fraction of the diagonal maximum.
        Default 0.02. Set to 0 to disable the prominence filter.
    **kwargs
        Additional keyword arguments passed through to `scipy.signal.find_peaks`
        (e.g. `height`, `width`, `distance`). Passing `prominence` directly
        overrides the value derived from `prominence_frac`.

    Returns
    -------
    AutopeakResult
        Named tuple of (positions, intensities). Empty arrays if none found.
    """
    phi = np.asarray(phi)
    wavenumbers = np.asarray(wavenumbers)

    if phi.ndim != 2:
        raise ValueError(f"phi must be 2D, got shape {phi.shape}")
    if phi.shape[0] != phi.shape[1]:
        raise ValueError(f"phi must be square, got shape {phi.shape}")
    if wavenumbers.ndim != 1:
        raise ValueError(f"wavenumbers must be 1D, got shape {wavenumbers.shape}")
    if wavenumbers.size != phi.shape[0]:
        raise ValueError(
            f"wavenumbers size ({wavenumbers.size}) does not match phi shape ({phi.shape})"
        )

    diagonal = np.diag(phi)

    # User-supplied absolute prominence wins over prominence_frac
    kwargs.setdefault("prominence", prominence_frac * diagonal.max())

    indices, _ = find_peaks(diagonal, **kwargs)

    return AutopeakResult(
        positions=wavenumbers[indices],
        intensities=diagonal[indices],
    )


def synchronous_hetero(
    series_1: SpectralSeries,
    series_2: SpectralSeries,
) -> np.ndarray:
    """Heterospectral synchronous 2DCOS matrix.

    Computes Φ(ν₁, ν₂) = Y_1^T @ Y_2 / (m - 1), where Y_1 and Y_2 are the
    mean-centred intensities of the two series. The two series must share
    the same perturbation array (rows are paired by index).

    Parameters
    ----------
    series_1
        SpectralSeries supplying ν₁ — maps to matrix axis 0 (rows), and to
        the y-axis when the result is plotted.
    series_2
        SpectralSeries supplying ν₂ — maps to matrix axis 1 (cols), and to
        the x-axis when the result is plotted.

    Returns
    -------
    np.ndarray
        Shape `(n_1, n_2)`. Not symmetric in general.

    Raises
    ------
    ValueError
        If the two series have different perturbation arrays.
    """
    _check_paired_perturbations(series_1, series_2)
    m = series_1.n_perturbations
    Y_1 = series_1.intensities - series_1.intensities.mean(axis=0)
    Y_2 = series_2.intensities - series_2.intensities.mean(axis=0)
    return (Y_1.T @ Y_2) / (m - 1)


def asynchronous_hetero(
    series_1: SpectralSeries,
    series_2: SpectralSeries,
) -> np.ndarray:
    """Heterospectral asynchronous 2DCOS matrix.

    Computes Ψ(ν₁, ν₂) = Y_1^T @ N @ Y_2 / (m - 1), where N is the
    Hilbert-Noda matrix of shape (m, m).

    See `synchronous_hetero` for axis conventions.

    Raises
    ------
    ValueError
        If the two series have different perturbation arrays.
    """
    _check_paired_perturbations(series_1, series_2)
    m = series_1.n_perturbations
    Y_1 = series_1.intensities - series_1.intensities.mean(axis=0)
    Y_2 = series_2.intensities - series_2.intensities.mean(axis=0)
    N = hilbert_noda_matrix(m)
    return (Y_1.T @ N @ Y_2) / (m - 1)


def _check_paired_perturbations(
    series_1: SpectralSeries,
    series_2: SpectralSeries,
) -> None:
    """Raise if two series don't share identical perturbation arrays."""
    if not np.array_equal(series_1.perturbations, series_2.perturbations):
        raise ValueError(
            "series_1 and series_2 must have identical perturbation arrays; "
            f"got shapes {series_1.perturbations.shape} vs "
            f"{series_2.perturbations.shape}"
        )
