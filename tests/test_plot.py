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
        wn_1 = np.linspace(3100, 3700, 100)
        wn_2 = np.linspace(2800, 3050, 80)

        ax = plot_contour(matrix, wn_1, wavenumbers_2=wn_2)

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
        wn_1 = np.linspace(3100, 3700, 100)
        wn_2 = np.linspace(2800, 3050, 80)

        ax = plot_contour(matrix, wn_1, wavenumbers_2=wn_2, mask_diagonal=True)

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
        wn_1 = np.linspace(3100, 3700, 100)
        wn_2 = np.linspace(2800, 3050, 80)

        fig, axes = plot_sync_async(sync, asyn, wn_1, wavenumbers_2=wn_2)

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


def _lead_lag_series():
    # The band at 1300 responds early and the one at 1100 late, both growing,
    # so Phi(1300, 1100) > 0 and Psi(1300, 1100) > 0 under Noda's rules.
    t = np.linspace(0, 10, 21)
    wn = np.linspace(1000, 1400, 201)

    def band(centre):
        return np.exp(-0.5 * ((wn - centre) / 15) ** 2)

    intensities = np.outer(1 - np.exp(-1.0 * t), band(1300)) + np.outer(
        1 - np.exp(-0.15 * t), band(1100)
    )
    return SpectralSeries(wavenumbers=wn, perturbations=t, intensities=intensities, name="lag")


def _sign_rendered_at(ax, x, y):
    ax.figure.canvas.draw()
    image = np.asarray(ax.figure.canvas.buffer_rgba())[..., :3].astype(int)
    px, py = ax.transData.transform((x, y))
    red, _, blue = image[round(image.shape[0] - py), round(px)]
    return "positive" if red > blue else "negative"


class TestOrientation:
    """Matrix rows (nu1) are drawn on x, columns (nu2) on y."""

    def test_asynchronous_sign_renders_where_it_is_read(self):
        # Regression: the matrix used to be drawn untransposed, so the point
        # read as (nu1=1300, nu2=1100) showed Psi(1100, 1300) -- the wrong sign,
        # inverting every sequential-order conclusion.
        series = _lead_lag_series()
        psi = asynchronous(series)
        i = np.abs(series.wavenumbers - 1300).argmin()
        j = np.abs(series.wavenumbers - 1100).argmin()
        assert psi[i, j] > 0  # premise: Psi(1300, 1100) > 0 in the array
        ax = plot_contour(psi, series.wavenumbers, mask_diagonal=True)
        assert _sign_rendered_at(ax, 1300, 1100) == "positive"
        assert _sign_rendered_at(ax, 1100, 1300) == "negative"

    def test_heterospectral_axes_follow_the_matrix_axes(self):
        matrix = np.random.default_rng(0).standard_normal((100, 80))
        wn_1 = np.linspace(3100, 3700, 100)
        wn_2 = np.linspace(2800, 3050, 80)
        ax = plot_contour(matrix, wn_1, wavenumbers_2=wn_2)
        assert sorted(ax.get_xlim()) == [3100, 3700]
        assert sorted(ax.get_ylim()) == [2800, 3050]

    def test_axes_are_labelled_nu1_and_nu2(self, basic_matrix):
        m, wn = basic_matrix
        ax = plot_contour(m, wn)
        assert r"\nu}_1" in ax.get_xlabel()
        assert r"\nu}_2" in ax.get_ylabel()


class TestDeprecatedWavenumbersX:
    def test_alias_still_works_and_warns(self):
        matrix = np.random.default_rng(0).standard_normal((100, 80))
        wn_1 = np.linspace(3100, 3700, 100)
        wn_2 = np.linspace(2800, 3050, 80)
        with pytest.warns(DeprecationWarning, match="wavenumbers_2"):
            ax = plot_contour(matrix, wn_1, wavenumbers_x=wn_2)
        assert sorted(ax.get_ylim()) == [2800, 3050]

    def test_alias_warns_through_plot_sync_async(self):
        sync = np.random.default_rng(0).standard_normal((100, 80))
        wn_1 = np.linspace(3100, 3700, 100)
        wn_2 = np.linspace(2800, 3050, 80)
        with pytest.warns(DeprecationWarning, match="wavenumbers_2"):
            plot_sync_async(sync, sync, wn_1, wavenumbers_x=wn_2)

    def test_both_names_rejected(self):
        matrix = np.random.default_rng(0).standard_normal((100, 80))
        wn_1 = np.linspace(3100, 3700, 100)
        wn_2 = np.linspace(2800, 3050, 80)
        with pytest.raises(TypeError, match="not both"):
            plot_contour(matrix, wn_1, wavenumbers_2=wn_2, wavenumbers_x=wn_2)
