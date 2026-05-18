from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from crosspeak import SpectralSeries


@pytest.fixture
def basic_series():
    rng = np.random.default_rng(0)
    return SpectralSeries(
        wavenumbers=np.linspace(3700, 3100, 50),  # descending, FTIR-typical
        perturbations=np.array([0, 1, 2, 5, 10]),
        intensities=rng.normal(size=(5, 50)),
        name="test",
    )


def test_construction_from_arrays(basic_series):
    assert basic_series.shape == (5, 50)
    assert basic_series.n_perturbations == 5
    assert basic_series.n_wavenumbers == 50
    assert basic_series.name == "test"


def test_construction_from_lists():
    s = SpectralSeries(
        wavenumbers=[3700, 3500, 3300, 3100],
        perturbations=[0, 1, 2],
        intensities=[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]],
    )
    assert s.shape == (3, 4)
    assert isinstance(s.wavenumbers, np.ndarray)
    assert isinstance(s.intensities, np.ndarray)


def test_name_defaults_to_none():
    s = SpectralSeries(
        wavenumbers=[1.0, 2.0, 3.0],
        perturbations=[0.0, 1.0],
        intensities=[[1, 2, 3], [4, 5, 6]],
    )
    assert s.name is None


def test_ascending_wavenumbers_ok():
    s = SpectralSeries(
        wavenumbers=[3100, 3300, 3500, 3700],
        perturbations=[0, 1, 2],
        intensities=np.zeros((3, 4)),
    )
    assert s.wavenumbers[0] < s.wavenumbers[-1]


def test_descending_wavenumbers_ok():
    s = SpectralSeries(
        wavenumbers=[3700, 3500, 3300, 3100],
        perturbations=[0, 1, 2],
        intensities=np.zeros((3, 4)),
    )
    assert s.wavenumbers[0] > s.wavenumbers[-1]


@pytest.mark.parametrize(
    "wn, pt, it, match",
    [
        # intensities not 2D
        ([1, 2, 3], [0, 1], [1, 2, 3, 4, 5, 6], "must be 2D"),
        # wavenumbers not 1D
        ([[1, 2], [3, 4]], [0, 1], [[1, 2], [3, 4]], "must be 1D"),
        # shape mismatch
        ([1, 2, 3], [0, 1], [[1, 2], [3, 4]], "does not match"),
        # non-monotonic wavenumbers (repeat)
        ([1, 1, 2], [0, 1], [[1, 1, 2], [3, 3, 4]], "strictly monotonic"),
        # non-monotonic perturbations (repeat)
        ([1, 2, 3], [0, 0], [[1, 2, 3], [4, 5, 6]], "strictly monotonic"),
        # too few wavenumbers
        ([1], [0, 1], [[1], [2]], "at least 2 wavenumber"),
        # too few perturbations
        ([1, 2], [0], [[1, 2]], "at least 2 perturbation"),
    ],
)
def test_invalid_inputs_raise(wn, pt, it, match):
    with pytest.raises(ValueError, match=match):
        SpectralSeries(wavenumbers=wn, perturbations=pt, intensities=it)


def test_attribute_assignment_blocked(basic_series):
    with pytest.raises(FrozenInstanceError):
        basic_series.name = "renamed"


def test_intensity_mutation_blocked(basic_series):
    with pytest.raises(ValueError, match="read-only"):
        basic_series.intensities[0, 0] = 999.0


def test_caller_arrays_not_mutated():
    wn = np.array([1.0, 2.0, 3.0])
    pt = np.array([0.0, 1.0])
    it = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    SpectralSeries(wavenumbers=wn, perturbations=pt, intensities=it)

    # caller's arrays remain writeable
    assert wn.flags.writeable
    assert pt.flags.writeable
    assert it.flags.writeable


def test_repr_contains_dimensions_and_name(basic_series):
    r = repr(basic_series)
    assert "5" in r and "50" in r
    assert "test" in r
