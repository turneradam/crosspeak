import numpy as np
import pytest

from crosspeak import SpectralSeries
from crosspeak.preprocess import (
    area_normalize,
    crop_region,
    mean_center,
    reference_spectrum,
    savgol_smooth,
)


@pytest.fixture
def simple_series():
    return SpectralSeries(
        wavenumbers=[1.0, 2.0, 3.0, 4.0],
        perturbations=[0, 1, 2],
        intensities=[
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ],
    )


def test_reference_is_column_mean(simple_series):
    ref = reference_spectrum(simple_series)
    np.testing.assert_allclose(ref, [2.0, 3.0, 4.0, 5.0])


def test_reference_shape(simple_series):
    ref = reference_spectrum(simple_series)
    assert ref.ndim == 1
    assert ref.size == simple_series.n_wavenumbers


def test_mean_center_returns_spectral_series(simple_series):
    out = mean_center(simple_series)
    assert isinstance(out, SpectralSeries)


def test_mean_center_shape_preserved(simple_series):
    out = mean_center(simple_series)
    assert out.shape == simple_series.shape


def test_mean_center_axes_preserved(simple_series):
    out = mean_center(simple_series)
    np.testing.assert_array_equal(out.wavenumbers, simple_series.wavenumbers)
    np.testing.assert_array_equal(out.perturbations, simple_series.perturbations)


def test_mean_center_name_passed_through():
    s = SpectralSeries(
        wavenumbers=[1.0, 2.0, 3.0],
        perturbations=[0, 1],
        intensities=[[1, 2, 3], [4, 5, 6]],
        name="UR33W",
    )
    out = mean_center(s)
    assert out.name == "UR33W"


def test_mean_center_produces_zero_column_sums(simple_series):
    out = mean_center(simple_series)
    column_sums = out.intensities.sum(axis=0)
    np.testing.assert_allclose(column_sums, np.zeros(simple_series.n_wavenumbers), atol=1e-12)


def test_mean_center_values_known(simple_series):
    expected = np.array(
        [
            [-1, -1, -1, -1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    out = mean_center(simple_series)
    np.testing.assert_allclose(out.intensities, expected)


def test_mean_center_does_not_mutate_input(simple_series):
    original = np.array(simple_series.intensities, copy=True)
    _ = mean_center(simple_series)
    np.testing.assert_array_equal(simple_series.intensities, original)


class TestCropRegion:
    def _make_series(self, wavenumbers, n_perturbations=4):
        """Build a SpectralSeries with predictable intensities for testing."""
        perturbations = np.arange(n_perturbations, dtype=float)
        # intensity[i, j] = i + j/1000 — uniquely identifies every cell
        intensities = perturbations[:, None] + wavenumbers[None, :] / 1000.0
        return SpectralSeries(
            wavenumbers=wavenumbers,
            perturbations=perturbations,
            intensities=intensities,
            name="test",
        )

    def test_ascending_basic_crop(self):
        wn = np.linspace(2000, 4000, 201)  # step = 10 cm-1
        series = self._make_series(wn)

        cropped = crop_region(series, 3000, 3500)

        assert cropped.wavenumbers.min() >= 3000
        assert cropped.wavenumbers.max() <= 3500
        assert cropped.wavenumbers[0] == 3000
        assert cropped.wavenumbers[-1] == 3500

    def test_descending_preserves_direction(self):
        wn = np.linspace(4000, 2000, 201)  # descending
        series = self._make_series(wn)

        cropped = crop_region(series, 3000, 3500)

        assert cropped.wavenumbers[0] > cropped.wavenumbers[-1]  # still descending
        assert cropped.wavenumbers.min() >= 3000
        assert cropped.wavenumbers.max() <= 3500

    def test_boundaries_inclusive(self):
        wn = np.array([2900.0, 3000.0, 3100.0, 3200.0, 3300.0])
        series = self._make_series(wn)

        cropped = crop_region(series, 3000.0, 3200.0)

        np.testing.assert_array_equal(cropped.wavenumbers, [3000.0, 3100.0, 3200.0])

    def test_partial_overlap_clips_silently(self):
        wn = np.linspace(3000, 3500, 51)
        series = self._make_series(wn)

        # Request extends beyond data on both sides — clip, don't raise
        cropped = crop_region(series, 2000, 4000)

        np.testing.assert_array_equal(cropped.wavenumbers, wn)

    def test_no_overlap_raises(self):
        wn = np.linspace(3000, 3500, 51)
        series = self._make_series(wn)

        with pytest.raises(ValueError, match="no overlap"):
            crop_region(series, 1000, 2000)

    def test_inverted_bounds_raises(self):
        wn = np.linspace(3000, 3500, 51)
        series = self._make_series(wn)

        with pytest.raises(ValueError, match="must be <="):
            crop_region(series, 3500, 3000)

    def test_preserves_perturbations_and_name(self):
        wn = np.linspace(2000, 4000, 201)
        series = self._make_series(wn, n_perturbations=5)

        cropped = crop_region(series, 3000, 3500)

        np.testing.assert_array_equal(cropped.perturbations, series.perturbations)
        assert cropped.name == series.name

    def test_intensities_sliced_correctly(self):
        wn = np.linspace(2000, 4000, 201)
        series = self._make_series(wn)
        mask = (wn >= 3000) & (wn <= 3500)
        expected = series.intensities[:, mask]

        cropped = crop_region(series, 3000, 3500)

        np.testing.assert_array_equal(cropped.intensities, expected)

    def test_original_series_unchanged(self):
        wn = np.linspace(2000, 4000, 201)
        series = self._make_series(wn)
        original_wn = series.wavenumbers.copy()
        original_intensities = series.intensities.copy()

        _ = crop_region(series, 3000, 3500)

        np.testing.assert_array_equal(series.wavenumbers, original_wn)
        np.testing.assert_array_equal(series.intensities, original_intensities)

    def test_pipeline_crop_then_synchronous(self):
        """End-to-end: crop_region → synchronous gives sensible shape."""
        from crosspeak import synchronous

        wn = np.linspace(2000, 4000, 401)
        series = self._make_series(wn, n_perturbations=5)
        cropped = crop_region(series, 3000, 3700)

        phi = synchronous(cropped)

        n_cropped = cropped.n_wavenumbers
        assert phi.shape == (n_cropped, n_cropped)
        assert n_cropped < series.n_wavenumbers  # actually cropped down


class TestSavgolSmooth:
    def _noisy_gaussian_series(self, n_pert=5, n_wn=401, sigma=30.0, seed=0):
        """Series of identical Gaussians with additive white noise."""
        rng = np.random.default_rng(seed)
        wn = np.linspace(2800, 3700, n_wn)
        perturbations = np.arange(n_pert, dtype=float)
        peak_idx = n_wn // 2
        # Gaussian centred mid-axis, amplitude scales with perturbation
        base = np.exp(-((np.arange(n_wn) - peak_idx) ** 2) / (2 * sigma**2))
        clean = perturbations[:, None] * base[None, :]
        noisy = clean + rng.normal(scale=0.02, size=clean.shape)
        return SpectralSeries(
            wavenumbers=wn,
            perturbations=perturbations,
            intensities=noisy,
            name="noisy",
        ), clean

    def test_returns_same_shape(self):
        series, _ = self._noisy_gaussian_series()
        smoothed = savgol_smooth(series)
        assert smoothed.intensities.shape == series.intensities.shape

    def test_reduces_noise(self):
        series, clean = self._noisy_gaussian_series()

        # Residual std vs. truth should drop after smoothing
        noisy_residual_std = np.std(series.intensities - clean)
        smoothed = savgol_smooth(series)
        smoothed_residual_std = np.std(smoothed.intensities - clean)

        assert smoothed_residual_std < noisy_residual_std

    def test_preserves_peak_height(self):
        # Wide-ish Gaussian (sigma=30 samples) with window=13 should retain >95% of height
        series, _ = self._noisy_gaussian_series(sigma=30.0)
        smoothed = savgol_smooth(series)

        # Compare max amplitude in highest-perturbation row vs the same row in input
        original_max = series.intensities[-1].max()
        smoothed_max = smoothed.intensities[-1].max()
        assert smoothed_max > 0.95 * original_max

    def test_preserves_metadata(self):
        series, _ = self._noisy_gaussian_series()
        smoothed = savgol_smooth(series)

        np.testing.assert_array_equal(smoothed.wavenumbers, series.wavenumbers)
        np.testing.assert_array_equal(smoothed.perturbations, series.perturbations)
        assert smoothed.name == series.name

    def test_original_unchanged(self):
        series, _ = self._noisy_gaussian_series()
        original_intensities = series.intensities.copy()

        _ = savgol_smooth(series)

        np.testing.assert_array_equal(series.intensities, original_intensities)

    def test_even_window_length_raises(self):
        series, _ = self._noisy_gaussian_series()
        with pytest.raises(ValueError, match="must be odd"):
            savgol_smooth(series, window_length=12)  # even -> scipy raises

    def test_kwargs_passthrough_deriv(self):
        """deriv=1 via kwargs should give a derivative, not a smoothed spectrum."""
        series, _ = self._noisy_gaussian_series()

        smoothed = savgol_smooth(series)
        derivative = savgol_smooth(series, deriv=1)

        # Derivative output should differ substantially from smoothed output
        assert not np.allclose(smoothed.intensities, derivative.intensities)
        # Smoothed Gaussian stays positive; its derivative crosses zero
        assert (derivative.intensities < 0).any()
        assert (derivative.intensities > 0).any()

    def test_pipeline_smooth_then_synchronous(self):
        from crosspeak import synchronous

        series, _ = self._noisy_gaussian_series()
        smoothed = savgol_smooth(series)
        phi = synchronous(smoothed)

        assert phi.shape == (series.n_wavenumbers, series.n_wavenumbers)
        # Diagonal should be non-negative (it's the autopower)
        assert (np.diag(phi) >= 0).all()


class TestAreaNormalize:
    def _series(self, wn=None, intensities=None, perturbations=None, name="test"):
        """Helper: Gaussian-band series with sensible defaults."""
        if wn is None:
            wn = np.linspace(2800, 3700, 200)
        if perturbations is None:
            perturbations = np.arange(4, dtype=float)
        if intensities is None:
            base = np.exp(-((wn - 3300) ** 2) / (2 * 50**2))
            intensities = (perturbations + 1)[:, None] * base[None, :]
        return SpectralSeries(
            wavenumbers=wn,
            perturbations=perturbations,
            intensities=intensities,
            name=name,
        )

    def test_unit_area_default(self):
        series = self._series()
        normalized = area_normalize(series)

        areas = np.abs(np.trapezoid(normalized.intensities, x=normalized.wavenumbers, axis=-1))
        np.testing.assert_allclose(areas, 1.0, atol=1e-10)

    def test_custom_target_area(self):
        series = self._series()
        normalized = area_normalize(series, target_area=10.0)

        areas = np.abs(np.trapezoid(normalized.intensities, x=normalized.wavenumbers, axis=-1))
        np.testing.assert_allclose(areas, 10.0, atol=1e-9)

    def test_metadata_preserved(self):
        series = self._series()
        normalized = area_normalize(series)

        np.testing.assert_array_equal(normalized.wavenumbers, series.wavenumbers)
        np.testing.assert_array_equal(normalized.perturbations, series.perturbations)
        assert normalized.name == series.name

    def test_original_unchanged(self):
        series = self._series()
        original = series.intensities.copy()

        _ = area_normalize(series)

        np.testing.assert_array_equal(series.intensities, original)

    def test_zero_area_raises(self):
        wn = np.linspace(2800, 3700, 200)
        perturbations = np.arange(3, dtype=float)
        intensities = np.ones((3, 200))
        intensities[1] = 0.0  # middle row is all zeros
        series = SpectralSeries(
            wavenumbers=wn,
            perturbations=perturbations,
            intensities=intensities,
            name="test",
        )

        with pytest.raises(ValueError, match="zero integrated area"):
            area_normalize(series)

    def test_descending_wavenumbers(self):
        wn = np.linspace(3700, 2800, 200)
        series = self._series(wn=wn)
        normalized = area_normalize(series)

        areas = np.abs(np.trapezoid(normalized.intensities, x=normalized.wavenumbers, axis=-1))
        np.testing.assert_allclose(areas, 1.0, atol=1e-10)

    def test_reference_region_normalizes_to_band(self):
        series = self._series()
        normalized = area_normalize(series, reference_region=(3000, 3500))

        mask = (normalized.wavenumbers >= 3000) & (normalized.wavenumbers <= 3500)
        ref_areas = np.abs(
            np.trapezoid(
                normalized.intensities[:, mask],
                x=normalized.wavenumbers[mask],
                axis=-1,
            )
        )
        np.testing.assert_allclose(ref_areas, 1.0, atol=1e-10)

    def test_reference_region_invalid_raises(self):
        series = self._series()

        with pytest.raises(ValueError, match="must be <="):
            area_normalize(series, reference_region=(3500, 3000))
        with pytest.raises(ValueError, match="no overlap"):
            area_normalize(series, reference_region=(1000, 2000))

    def test_pipeline_normalize_then_synchronous(self):
        from crosspeak import synchronous

        series = self._series()
        normalized = area_normalize(series)
        phi = synchronous(normalized)

        assert phi.shape == (series.n_wavenumbers, series.n_wavenumbers)
        assert (np.diag(phi) >= 0).all()
