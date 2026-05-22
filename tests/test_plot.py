import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from crosspeak import (
    SpectralSeries,
    asynchronous,
    plot_contour,
    synchronous,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def basic_matrix():
    n = 50
    wn = np.linspace(3700, 3100, n)
    rng = np.random.default_rng(0)
    m = rng.normal(size=(n, n))
    m = (m + m.T) / 2
    return m, wn


def test_returns_axes(basic_matrix):
    matrix, wn = basic_matrix
    ax = plot_contour(matrix, wn)
    assert isinstance(ax, Axes)


def test_uses_provided_axes(basic_matrix):
    matrix, wn = basic_matrix
    _, ax_in = plt.subplots()
    ax_out = plot_contour(matrix, wn, ax=ax_in)
    assert ax_out is ax_in


def test_title_set(basic_matrix):
    matrix, wn = basic_matrix
    ax = plot_contour(matrix, wn, title="GY33W synchronous")
    assert ax.get_title() == "GY33W synchronous"


def test_descending_axes_default(basic_matrix):
    matrix, wn = basic_matrix
    ax = plot_contour(matrix, wn)
    assert ax.get_xlim()[0] > ax.get_xlim()[1]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]


def test_ascending_axes(basic_matrix):
    matrix, wn = basic_matrix
    ax = plot_contour(matrix, wn, descending=False)
    assert ax.get_xlim()[0] < ax.get_xlim()[1]
    assert ax.get_ylim()[0] < ax.get_ylim()[1]


def test_rejects_non_square_matrix(basic_matrix):
    _, wn = basic_matrix
    matrix = np.zeros((5, 6))
    with pytest.raises(ValueError, match="square"):
        plot_contour(matrix, wn[:5])


def test_rejects_axis_mismatch(basic_matrix):
    matrix, wn = basic_matrix
    with pytest.raises(ValueError, match="doesn't match"):
        plot_contour(matrix, wn[:10])


def test_vlag_registered():
    assert "vlag" in plt.colormaps()


def test_all_zero_matrix_does_not_crash(basic_matrix):
    _, wn = basic_matrix
    matrix = np.zeros_like(basic_matrix[0])
    ax = plot_contour(matrix, wn)
    assert isinstance(ax, Axes)


def test_full_pipeline_synchronous():
    rng = np.random.default_rng(0)
    wn = np.linspace(3700, 3100, 100)
    intensities = rng.normal(size=(5, 100))
    s = SpectralSeries(
        wavenumbers=wn,
        perturbations=[0, 1, 2, 3, 4],
        intensities=intensities,
        name="test",
    )
    phi = synchronous(s)
    ax = plot_contour(phi, s.wavenumbers, title="test sync")
    assert ax.get_title() == "test sync"


def test_full_pipeline_asynchronous():
    rng = np.random.default_rng(0)
    wn = np.linspace(3700, 3100, 100)
    intensities = rng.normal(size=(5, 100))
    s = SpectralSeries(
        wavenumbers=wn,
        perturbations=[0, 1, 2, 3, 4],
        intensities=intensities,
        name="test",
    )
    psi = asynchronous(s)
    ax = plot_contour(psi, s.wavenumbers, title="test async")
    assert ax.get_title() == "test async"
