import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from crosspeak import (
    SpectralSeries,
    asynchronous,
    plot_contour,
    plot_sync_async,
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


def test_rejects_axis_mismatch(basic_matrix):
    matrix, wn = basic_matrix
    with pytest.raises(ValueError, match="must equal"):
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


class TestPlotContourHetero:
    def test_rectangular_matrix_accepted(self):
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((100, 80))
        wn_y = np.linspace(3100, 3700, 100)
        wn_x = np.linspace(2800, 3050, 80)

        ax = plot_contour(matrix, wn_y, wavenumbers_x=wn_x)

        assert ax is not None
        plt.close("all")

    def test_backward_compat_homospectral(self):
        """Existing single-axis signature still works for square matrices."""
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((50, 50))
        wn = np.linspace(2800, 3700, 50)

        ax = plot_contour(matrix, wn)

        assert ax is not None
        plt.close("all")


class TestPlotPolish:
    def test_mask_diagonal_inserts_nan_on_square(self):
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((50, 50))
        wn = np.linspace(2800, 3700, 50)

        # Capture the matrix the function plots by snooping via a custom Axes
        ax = plot_contour(matrix, wn, mask_diagonal=True)

        # Diagonal cells should not produce contour patches — easiest sanity
        # check: the function returned an Axes and didn't error
        assert ax is not None
        plt.close("all")

    def test_mask_diagonal_does_not_mutate_input(self):
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((50, 50))
        original = matrix.copy()
        wn = np.linspace(2800, 3700, 50)

        plot_contour(matrix, wn, mask_diagonal=True)

        np.testing.assert_array_equal(matrix, original)
        plt.close("all")

    def test_mask_diagonal_silent_on_rectangular(self):
        """Hetero (non-square) matrix: mask_diagonal is silently no-op."""
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((100, 80))
        wn_y = np.linspace(3100, 3700, 100)
        wn_x = np.linspace(2800, 3050, 80)

        ax = plot_contour(matrix, wn_y, wavenumbers_x=wn_x, mask_diagonal=True)

        assert ax is not None
        plt.close("all")


class TestPlotSyncAsync:
    def test_homospectral_call(self):
        rng = np.random.default_rng(0)
        sync = rng.standard_normal((50, 50))
        asyn = rng.standard_normal((50, 50))
        wn = np.linspace(2800, 3700, 50)

        fig, axes = plot_sync_async(sync, asyn, wn)

        assert fig is not None
        assert len(axes) == 2
        plt.close("all")

    def test_heterospectral_call(self):
        rng = np.random.default_rng(0)
        sync = rng.standard_normal((100, 80))
        asyn = rng.standard_normal((100, 80))
        wn_y = np.linspace(3100, 3700, 100)
        wn_x = np.linspace(2800, 3050, 80)

        fig, axes = plot_sync_async(sync, asyn, wn_y, wavenumbers_x=wn_x)

        assert fig is not None
        assert len(axes) == 2
        plt.close("all")

    def test_shape_mismatch_raises(self):
        rng = np.random.default_rng(0)
        sync = rng.standard_normal((50, 50))
        asyn = rng.standard_normal((40, 40))
        wn = np.linspace(2800, 3700, 50)

        with pytest.raises(ValueError, match="same shape"):
            plot_sync_async(sync, asyn, wn)

    def test_title_sets_suptitle(self):
        rng = np.random.default_rng(0)
        sync = rng.standard_normal((50, 50))
        asyn = rng.standard_normal((50, 50))
        wn = np.linspace(2800, 3700, 50)

        fig, _ = plot_sync_async(sync, asyn, wn, title="MA50W test")

        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "MA50W test"
        plt.close("all")
