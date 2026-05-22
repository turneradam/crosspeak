import numpy as np
import pytest

from crosspeak import SpectralSeries, asynchronous, hilbert_noda_matrix, synchronous


def test_shape():
    N = hilbert_noda_matrix(5)
    assert N.shape == (5, 5)


def test_zero_diagonal():
    N = hilbert_noda_matrix(7)
    np.testing.assert_array_equal(np.diag(N), np.zeros(7))


def test_antisymmetric():
    N = hilbert_noda_matrix(6)
    np.testing.assert_allclose(N, -N.T)


def test_specific_values_n3():
    N = hilbert_noda_matrix(3)
    expected = np.array(
        [
            [0.0, 1 / np.pi, 1 / (2 * np.pi)],
            [-1 / np.pi, 0.0, 1 / np.pi],
            [-1 / (2 * np.pi), -1 / np.pi, 0.0],
        ]
    )
    np.testing.assert_allclose(N, expected)


def test_off_diagonal_pattern():
    N = hilbert_noda_matrix(5)
    for k in range(1, 5):
        assert np.isclose(N[0, k], 1.0 / (np.pi * k))
        assert np.isclose(N[k, 0], -1.0 / (np.pi * k))


def test_minimum_size():
    N = hilbert_noda_matrix(2)
    assert N.shape == (2, 2)


def test_too_small_raises():
    with pytest.raises(ValueError, match="at least 2"):
        hilbert_noda_matrix(1)


def test_non_integer_raises():
    with pytest.raises(TypeError, match="must be an integer"):
        hilbert_noda_matrix(3.5)


def _gauss(wn, center, width=20.0, amp=1.0):
    return amp * np.exp(-0.5 * ((wn - center) / width) ** 2)


@pytest.fixture
def correlated_series():
    """Two peaks at 3400 and 3200, both growing as perturbation increases."""
    wn = np.linspace(3700, 3100, 200)
    perturbations = np.array([0, 1, 2, 3, 4])

    intensities = np.empty((5, 200))
    for k, p in enumerate(perturbations):
        peak_a = _gauss(wn, 3400, amp=1.0 + 0.5 * p)
        peak_b = _gauss(wn, 3200, amp=1.0 + 0.5 * p)
        intensities[k] = peak_a + peak_b

    return SpectralSeries(
        wavenumbers=wn,
        perturbations=perturbations,
        intensities=intensities,
        name="correlated",
    )


def test_synchronous_shape(correlated_series):
    phi = synchronous(correlated_series)
    n = correlated_series.n_wavenumbers
    assert phi.shape == (n, n)


def test_synchronous_symmetric(correlated_series):
    phi = synchronous(correlated_series)
    np.testing.assert_allclose(phi, phi.T)


def test_synchronous_diagonal_nonnegative(correlated_series):
    phi = synchronous(correlated_series)
    assert np.all(np.diag(phi) >= 0)


def test_synchronous_correlated_bands_positive(correlated_series):
    phi = synchronous(correlated_series)
    wn = correlated_series.wavenumbers
    i = np.argmin(np.abs(wn - 3400))
    j = np.argmin(np.abs(wn - 3200))
    assert phi[i, j] > 0


def test_synchronous_anticorrelated_bands_negative():
    wn = np.linspace(3700, 3100, 200)
    perturbations = np.array([0, 1, 2, 3, 4])

    intensities = np.empty((5, 200))
    for k, p in enumerate(perturbations):
        peak_grows = _gauss(wn, 3400, amp=1.0 + 0.5 * p)
        peak_shrinks = _gauss(wn, 3200, amp=2.0 - 0.5 * p)
        intensities[k] = peak_grows + peak_shrinks

    s = SpectralSeries(
        wavenumbers=wn,
        perturbations=perturbations,
        intensities=intensities,
    )
    phi = synchronous(s)

    i = np.argmin(np.abs(wn - 3400))
    j = np.argmin(np.abs(wn - 3200))
    assert phi[i, j] < 0


def test_synchronous_no_change_gives_zero():
    wn = np.linspace(3700, 3100, 100)
    base = _gauss(wn, 3400)
    intensities = np.tile(base[None, :], (5, 1))

    s = SpectralSeries(
        wavenumbers=wn,
        perturbations=[0, 1, 2, 3, 4],
        intensities=intensities,
    )
    phi = synchronous(s)
    np.testing.assert_allclose(phi, np.zeros_like(phi), atol=1e-12)


def test_synchronous_rejects_non_series():
    with pytest.raises(TypeError, match="SpectralSeries"):
        synchronous(np.zeros((5, 100)))


@pytest.fixture
def sequential_series():
    """Peak at 3400 changes early (p=0->2), peak at 3200 changes later (p=2->4)."""
    wn = np.linspace(3700, 3100, 200)
    perturbations = np.array([0, 1, 2, 3, 4])

    a_amp = [0.0, 1.0, 2.0, 2.0, 2.0]
    b_amp = [0.0, 0.0, 0.0, 1.0, 2.0]

    intensities = np.empty((5, 200))
    for k in range(5):
        peak_a = _gauss(wn, 3400, amp=a_amp[k])
        peak_b = _gauss(wn, 3200, amp=b_amp[k])
        intensities[k] = peak_a + peak_b

    return SpectralSeries(
        wavenumbers=wn,
        perturbations=perturbations,
        intensities=intensities,
    )


def test_asynchronous_shape(correlated_series):
    psi = asynchronous(correlated_series)
    n = correlated_series.n_wavenumbers
    assert psi.shape == (n, n)


def test_asynchronous_antisymmetric(correlated_series):
    psi = asynchronous(correlated_series)
    np.testing.assert_allclose(psi, -psi.T, atol=1e-12)


def test_asynchronous_zero_diagonal(correlated_series):
    psi = asynchronous(correlated_series)
    np.testing.assert_allclose(np.diag(psi), np.zeros(correlated_series.n_wavenumbers), atol=1e-12)


def test_asynchronous_correlated_near_zero(correlated_series):
    psi = asynchronous(correlated_series)
    wn = correlated_series.wavenumbers
    i = np.argmin(np.abs(wn - 3400))
    j = np.argmin(np.abs(wn - 3200))
    # Both peaks change in lockstep → no time lag → Ψ should vanish
    assert abs(psi[i, j]) < 1e-10


def test_asynchronous_sequential_sign(sequential_series):
    psi = asynchronous(sequential_series)
    phi = synchronous(sequential_series)
    wn = sequential_series.wavenumbers
    i = np.argmin(np.abs(wn - 3400))  # peak A — changes first
    j = np.argmin(np.abs(wn - 3200))  # peak B — changes later

    # Sanity: both grew net positive, so Φ > 0
    assert phi[i, j] > 0
    # Noda's rule: Φ > 0 and i precedes j → Ψ(i, j) > 0
    assert psi[i, j] > 0


def test_asynchronous_no_change_gives_zero():
    wn = np.linspace(3700, 3100, 100)
    base = _gauss(wn, 3400)
    intensities = np.tile(base[None, :], (5, 1))

    s = SpectralSeries(
        wavenumbers=wn,
        perturbations=[0, 1, 2, 3, 4],
        intensities=intensities,
    )
    psi = asynchronous(s)
    np.testing.assert_allclose(psi, np.zeros_like(psi), atol=1e-12)


def test_asynchronous_rejects_non_series():
    with pytest.raises(TypeError, match="SpectralSeries"):
        asynchronous(np.zeros((5, 100)))
