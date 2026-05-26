import numpy as np
import pytest

from crosspeak import SpectralSeries
from crosspeak.preprocess import crop_region, mean_center, reference_spectrum


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
