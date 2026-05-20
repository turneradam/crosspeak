import numpy as np
import pytest

from crosspeak import SpectralSeries
from crosspeak.preprocess import mean_center, reference_spectrum


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
