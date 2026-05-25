import numpy as np
import pytest

from crosspeak import (
    AutopeakResult,
    SpectralSeries,
    asynchronous,
    find_autopeaks,
    hilbert_noda_matrix,
    synchronous,
)


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


def _diag_matrix(values):
    """Build a fake Phi with `values` on the diagonal, zeros elsewhere."""
    matrix = np.zeros((len(values), len(values)))
    np.fill_diagonal(matrix, values)
    return matrix


class TestFindAutopeaks:
    def test_single_peak_detected(self):
        wavenumbers = np.linspace(2800, 3700, 100)
        diagonal = np.exp(-((np.arange(100) - 50) ** 2) / 50)
        phi = _diag_matrix(diagonal)

        result = find_autopeaks(phi, wavenumbers)

        assert isinstance(result, AutopeakResult)
        assert result.positions.size == 1
        np.testing.assert_allclose(result.positions[0], wavenumbers[50])
        np.testing.assert_allclose(result.intensities[0], 1.0)

    def test_multiple_peaks_detected(self):
        wavenumbers = np.linspace(2800, 3700, 200)
        diagonal = (
            np.exp(-((np.arange(200) - 40) ** 2) / 50)
            + np.exp(-((np.arange(200) - 100) ** 2) / 50)
            + np.exp(-((np.arange(200) - 160) ** 2) / 50)
        )
        phi = _diag_matrix(diagonal)

        result = find_autopeaks(phi, wavenumbers)

        assert result.positions.size == 3
        np.testing.assert_allclose(result.positions, wavenumbers[[40, 100, 160]], atol=1e-10)

    def test_flat_diagonal_returns_empty(self):
        wavenumbers = np.linspace(2800, 3700, 50)
        phi = np.zeros((50, 50))

        result = find_autopeaks(phi, wavenumbers)

        assert result.positions.size == 0
        assert result.intensities.size == 0

    def test_prominence_threshold_filters_small_peaks(self):
        wavenumbers = np.linspace(2800, 3700, 200)
        diagonal = np.exp(-((np.arange(200) - 40) ** 2) / 50) + 0.01 * np.exp(
            -((np.arange(200) - 160) ** 2) / 50
        )
        phi = _diag_matrix(diagonal)

        result_default = find_autopeaks(phi, wavenumbers)
        result_loose = find_autopeaks(phi, wavenumbers, prominence_frac=0.001)

        assert result_default.positions.size == 1
        assert result_loose.positions.size == 2

    def test_explicit_prominence_kwarg_wins(self):
        wavenumbers = np.linspace(2800, 3700, 200)
        diagonal = np.exp(-((np.arange(200) - 100) ** 2) / 50)
        phi = _diag_matrix(diagonal)

        # Absolute prominence above the peak's prominence rejects it
        result = find_autopeaks(phi, wavenumbers, prominence=2.0)
        assert result.positions.size == 0

    def test_namedtuple_unpacks(self):
        wavenumbers = np.linspace(2800, 3700, 100)
        diagonal = np.exp(-((np.arange(100) - 50) ** 2) / 50)
        phi = _diag_matrix(diagonal)

        positions, intensities = find_autopeaks(phi, wavenumbers)

        np.testing.assert_allclose(positions, [wavenumbers[50]])
        np.testing.assert_allclose(intensities, [1.0])

    def test_descending_wavenumbers_preserved(self):
        wavenumbers = np.linspace(3700, 2800, 100)
        diagonal = np.exp(-((np.arange(100) - 30) ** 2) / 50)
        phi = _diag_matrix(diagonal)

        result = find_autopeaks(phi, wavenumbers)

        assert result.positions.size == 1
        np.testing.assert_allclose(result.positions[0], wavenumbers[30])

    def test_non_square_phi_raises(self):
        with pytest.raises(ValueError, match="square"):
            find_autopeaks(np.zeros((10, 20)), np.arange(10))

    def test_wrong_ndim_phi_raises(self):
        with pytest.raises(ValueError, match="2D"):
            find_autopeaks(np.zeros(10), np.arange(10))

    def test_wavenumber_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            find_autopeaks(np.zeros((10, 10)), np.arange(5))

    def test_pipeline_synchronous_then_autopeaks(self):
        """End-to-end: SpectralSeries → synchronous → find_autopeaks."""
        from crosspeak import SpectralSeries, synchronous

        wn = np.linspace(2800, 3700, 200)
        perturbations = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
        band = np.exp(-((wn - 3300) ** 2) / 50**2)
        intensities = np.outer(perturbations, band)
        series = SpectralSeries(
            wavenumbers=wn,
            perturbations=perturbations,
            intensities=intensities,
            name="test",
        )

        phi = synchronous(series)
        result = find_autopeaks(phi, wn)

        assert result.positions.size >= 1
        strongest = result.positions[np.argmax(result.intensities)]
        assert abs(strongest - 3300) < 10  # within 10 cm-1 of expected band centre
