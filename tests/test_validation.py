"""Validation of the 2DCOS core.

Two kinds of check live here.

Algebraic identities hold for any correct implementation and need no external
reference. They pin the code to the formalism.

Cross-implementation agreement compares crosspeak against corr2D (Geitner et
al., J. Stat. Softw. 90(3), 2019), an independent peer-reviewed implementation
written in R that computes the correlation matrix by FFT rather than by the
Hilbert-Noda matrix product used here. Agreement between two different
algorithms in two different languages is evidence a shared bug is implausible.

The reference matrices in fixtures/ were produced once by
fixtures/generate_reference.R; R is never invoked from the test suite.

Two conventions to be aware of, both established empirically against corr2D
1.0.3 and documented in the JOSS paper:

1. corr2D normalises with 1/(pi*(m-1)) and its FFT carries a further factor of
   m from Parseval's theorem, so Re(corr2D$FT) = (m/pi) * Phi_Noda. The
   committed fixtures are pre-scaled by pi/m.

2. corr2D's frequency selection omits the Nyquist term when m is even. Its
   synchronous matrix therefore equals Noda's only for odd m. The fixture uses
   m = 11.

3. The asynchronous matrices are NOT related by a scalar. The Hilbert-Noda
   matrix is a truncated convolution kernel; corr2D's FFT computes the exact
   DFT Hilbert transform. These are different discretisations of the same
   continuous quantity and differ by tens of percent at any m, without
   converging. Their SIGNS agree, and Noda's sequential-order rules depend only
   on signs, so the asynchronous check is a sign-agreement check.
"""

from pathlib import Path

import numpy as np
import pytest

from crosspeak import (
    SpectralSeries,
    asynchronous,
    asynchronous_hetero,
    hilbert_noda_matrix,
    synchronous,
    synchronous_hetero,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return np.loadtxt(FIXTURES / name, delimiter=",", skiprows=1)


def _series(intensities, name="fixture"):
    m, n = intensities.shape
    return SpectralSeries(
        wavenumbers=np.linspace(1000.0, 1400.0, n),
        perturbations=np.arange(m, dtype=float),
        intensities=intensities,
        name=name,
    )


@pytest.fixture(scope="module")
def sim2d():
    return _series(_load("sim2d_input.csv"), name="sim2ddata")


class TestAlgebraicIdentities:
    """Properties any correct implementation must satisfy."""

    def test_synchronous_is_symmetric(self, sim2d):
        phi = synchronous(sim2d)
        np.testing.assert_allclose(phi, phi.T, atol=1e-14)

    def test_asynchronous_is_antisymmetric(self, sim2d):
        psi = asynchronous(sim2d)
        np.testing.assert_allclose(psi, -psi.T, atol=1e-14)

    def test_asynchronous_diagonal_vanishes(self, sim2d):
        # No band can lag itself.
        np.testing.assert_allclose(np.diag(asynchronous(sim2d)), 0.0, atol=1e-14)

    def test_synchronous_diagonal_is_nonnegative(self, sim2d):
        # The diagonal is the autopower spectrum: a variance, hence >= 0.
        assert (np.diag(synchronous(sim2d)) >= 0).all()

    def test_synchronous_equals_covariance(self, sim2d):
        # Phi is the sample covariance of the intensities across perturbation.
        np.testing.assert_allclose(synchronous(sim2d), np.cov(sim2d.intensities.T), atol=1e-14)

    def test_hilbert_noda_is_antisymmetric_with_zero_diagonal(self):
        for m in (2, 3, 8, 11):
            N = hilbert_noda_matrix(m)
            np.testing.assert_allclose(N, -N.T, atol=1e-15)
            np.testing.assert_allclose(np.diag(N), 0.0, atol=1e-15)

    def test_hilbert_noda_entries(self):
        # N_jk = 1 / (pi (k - j)) off the diagonal.
        N = hilbert_noda_matrix(4)
        assert N[0, 1] == pytest.approx(1 / np.pi)
        assert N[1, 0] == pytest.approx(-1 / np.pi)
        assert N[0, 3] == pytest.approx(1 / (3 * np.pi))

    def test_asynchronous_vanishes_for_two_spectra(self):
        # With m = 2 the mean-centred rows are +d/2 and -d/2, and N has a single
        # off-diagonal magnitude. The contraction cancels identically, so no
        # sequential order can be inferred from two spectra. This is a sharp
        # check on the sign and scaling of N.
        rng = np.random.default_rng(0)
        for _ in range(5):
            series = _series(rng.standard_normal((2, 6)))
            np.testing.assert_allclose(asynchronous(series), 0.0, atol=1e-14)

    def test_heterospectral_with_self_reduces_to_homospectral(self, sim2d):
        np.testing.assert_allclose(synchronous_hetero(sim2d, sim2d), synchronous(sim2d), atol=1e-14)
        np.testing.assert_allclose(
            asynchronous_hetero(sim2d, sim2d), asynchronous(sim2d), atol=1e-14
        )

    def test_scaling_the_data_scales_phi_quadratically(self, sim2d):
        scaled = _series(sim2d.intensities * 10.0)
        np.testing.assert_allclose(synchronous(scaled), synchronous(sim2d) * 100.0, rtol=1e-12)


class TestNodaRules:
    """The sign conventions that carry all the physical interpretation."""

    def test_early_band_leads_late_band(self):
        # Band 1 responds quickly, band 2 slowly; both increase.
        # Noda: Phi > 0 and Psi > 0  =>  nu1 changes before nu2.
        t = np.linspace(0, 10, 40)
        intensities = np.column_stack([1 - np.exp(-1.0 * t), 1 - np.exp(-0.15 * t)])
        series = _series(intensities)
        assert synchronous(series)[0, 1] > 0
        assert asynchronous(series)[0, 1] > 0

    def test_anticorrelated_bands_give_negative_sync(self):
        t = np.linspace(0, 10, 40)
        intensities = np.column_stack([1 - np.exp(-t), np.exp(-t)])
        assert synchronous(_series(intensities))[0, 1] < 0


class TestAgainstCorr2D:
    """Cross-implementation agreement with corr2D 1.0.3 (R, FFT-based)."""

    def test_synchronous_matches_corr2d_exactly(self, sim2d):
        # m = 11 is odd, so corr2D's FFT retains every frequency component and
        # its synchronous matrix equals Noda's after the pi/m rescaling applied
        # in generate_reference.R. Agreement should be at float64 round-off.
        reference = _load("sim2d_sync_noda.csv")
        np.testing.assert_allclose(synchronous(sim2d), reference, rtol=1e-10, atol=1e-14)

    def test_asynchronous_sign_matches_corr2d(self, sim2d):
        # The magnitudes are not comparable (see module docstring), but every
        # significant cross-peak must carry the same sign, since that is what
        # Noda's sequential-order rules read.
        reference = _load("sim2d_async_corr2d_scaled.csv")
        psi = asynchronous(sim2d)
        significant = np.abs(reference) > 1e-9 * np.abs(reference).max()
        agreement = np.mean(np.sign(psi[significant]) == np.sign(reference[significant]))
        assert agreement == 1.0

    def test_asynchronous_magnitudes_are_not_expected_to_match(self, sim2d):
        # Guard against a future refactor silently switching crosspeak to an
        # FFT Hilbert transform. If this ever starts passing, the discretisation
        # changed and the docs must change with it.
        reference = _load("sim2d_async_corr2d_scaled.csv")
        assert not np.allclose(asynchronous(sim2d), reference, rtol=1e-6)
