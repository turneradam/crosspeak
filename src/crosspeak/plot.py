import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Approximation of seaborn's vlag diverging palette (deep blue → near-white → brick red).
# 7 control points, interpolated to 256 levels by matplotlib.
VLAG_STOPS = [
    (0.137, 0.412, 0.627),
    (0.55, 0.75, 0.86),
    (0.85, 0.92, 0.95),
    (0.97, 0.97, 0.97),
    (0.95, 0.85, 0.82),
    (0.90, 0.55, 0.50),
    (0.65, 0.20, 0.20),
]
VLAG_CMAP = LinearSegmentedColormap.from_list("vlag", VLAG_STOPS, N=256)


def _register_palettes():
    if "vlag" not in plt.colormaps():
        mpl.colormaps.register(VLAG_CMAP)


_register_palettes()


_NU1_LABEL = r"$\tilde{\nu}_1$ (cm$^{-1}$)"
_NU2_LABEL = r"$\tilde{\nu}_2$ (cm$^{-1}$)"


def plot_contour(
    matrix: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    wavenumbers_2: np.ndarray | None = None,
    wavenumbers_x: np.ndarray | None = None,
    ax=None,
    title: str | None = None,
    cmap: str = "vlag",
    n_levels: int = 50,
    descending: bool = True,
    mask_diagonal: bool = False,
):
    """
    Contour plot of a 2D correlation matrix.

    The matrix is indexed [ν₁, ν₂], as returned by `synchronous`, `asynchronous`
    and their heterospectral counterparts, and is drawn with ν₁ on the x-axis
    and ν₂ on the y-axis. That is the orientation Noda's rules are usually read
    in: for an asynchronous map, a positive peak at (x, y) alongside a positive
    synchronous peak means the band at x changes before the band at y. The axes
    are labelled ν₁ and ν₂ so the orientation is never left implicit.

    Parameters
    ----------
    matrix
        2D correlation matrix, shape (n_1, n_2).
    wavenumbers
        ν₁ axis, length n_1 (matrix rows). Drawn on the x-axis.
    wavenumbers_2
        ν₂ axis, length n_2 (matrix columns). Drawn on the y-axis. If None,
        `wavenumbers` is used for both axes — the homospectral case.
    wavenumbers_x
        Deprecated alias for `wavenumbers_2`. Despite its name it is the ν₂
        axis and is drawn on the y-axis.
    """

    if mask_diagonal and matrix.shape[0] == matrix.shape[1]:
        matrix = matrix.astype(float, copy=True)
        np.fill_diagonal(matrix, np.nan)

    wavenumbers_2 = _second_axis(wavenumbers_2, wavenumbers_x)
    homospectral = wavenumbers_2 is None

    matrix = np.asarray(matrix)
    wavenumbers = np.asarray(wavenumbers)
    wavenumbers_2 = wavenumbers if homospectral else np.asarray(wavenumbers_2)

    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
    if wavenumbers.ndim != 1 or wavenumbers_2.ndim != 1:
        raise ValueError("wavenumber axes must be 1D")
    if matrix.shape[0] != wavenumbers.size:
        raise ValueError(
            f"matrix.shape[0] ({matrix.shape[0]}) must equal wavenumbers size ({wavenumbers.size})"
        )
    if matrix.shape[1] != wavenumbers_2.size:
        raise ValueError(
            f"matrix.shape[1] ({matrix.shape[1]}) must equal "
            f"wavenumbers_2 size ({wavenumbers_2.size})"
        )

    if mask_diagonal and matrix.shape[0] == matrix.shape[1]:
        matrix = matrix.astype(float, copy=True)
        np.fill_diagonal(matrix, np.nan)
    if wavenumbers.ndim != 1:
        raise ValueError(f"wavenumbers must be 1D, got shape {wavenumbers.shape}")
    if matrix.shape[0] != wavenumbers.size:
        raise ValueError(
            f"matrix dimension {matrix.shape[0]} doesn't match "
            f"wavenumber axis length {wavenumbers.size}"
        )

    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    # Pin zero to the middle of the diverging colormap regardless of data asymmetry
    vmax = np.nanmax(np.abs(matrix))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # contourf places Z[i, j] at (x[j], y[i]), so the matrix is transposed to
    # put rows (ν₁) on x and columns (ν₂) on y.
    cf = ax.contourf(
        wavenumbers,
        wavenumbers_2,
        matrix.T,
        levels=n_levels,
        cmap=cmap,
        norm=norm,
    )

    ax.contour(
        wavenumbers,
        wavenumbers_2,
        matrix.T,
        levels=10,
        colors="black",
        linewidths=0.4,
        alpha=0.6,
    )

    if descending:
        ax.set_xlim(wavenumbers.max(), wavenumbers.min())
        ax.set_ylim(wavenumbers_2.max(), wavenumbers_2.min())
    else:
        ax.set_xlim(wavenumbers.min(), wavenumbers.max())
        ax.set_ylim(wavenumbers_2.min(), wavenumbers_2.max())

    ax.set_xlabel(_NU1_LABEL)
    ax.set_ylabel(_NU2_LABEL)
    ax.set_aspect("equal" if wavenumbers is wavenumbers_2 else "auto")

    if title:
        ax.set_title(title)

    plt.colorbar(cf, ax=ax, label="Correlation intensity")

    return ax


def plot_sync_async(
    sync: np.ndarray,
    asyn: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    wavenumbers_2: np.ndarray | None = None,
    wavenumbers_x: np.ndarray | None = None,
    title: str | None = None,
    cmap: str = "vlag",
    n_levels: int = 50,
    descending: bool = True,
    figsize: tuple[float, float] = (12, 5),
):
    """Plot synchronous and asynchronous 2DCOS matrices side-by-side.

    Convenience wrapper around two `plot_contour` calls. The asynchronous
    panel has its diagonal masked automatically. Figure uses constrained
    layout, so colorbars and titles don't crowd the plot area.

    Parameters
    ----------
    sync
        Synchronous correlation matrix.
    asyn
        Asynchronous correlation matrix. Must have the same shape as `sync`.
    wavenumbers
        ν₁ axis (matrix rows), drawn on the x-axis of both panels.
    wavenumbers_2
        ν₂ axis (matrix columns), drawn on the y-axis. If None, `wavenumbers`
        is used for both axes (homospectral). For heterospectral plots, supply
        the second series' wavenumbers here.
    wavenumbers_x
        Deprecated alias for `wavenumbers_2`.
    title
        Optional figure-level title (suptitle).
    cmap, n_levels, descending, figsize
        Passed through to `plot_contour` / `plt.subplots` as expected.

    Returns
    -------
    (fig, axes)
        `axes` is a length-2 numpy array of matplotlib Axes. Unpack as
        `fig, (ax_sync, ax_async) = plot_sync_async(...)` to reference each.

    Raises
    ------
    ValueError
        If `sync` and `asyn` have different shapes.
    """
    wavenumbers_2 = _second_axis(wavenumbers_2, wavenumbers_x)
    if sync.shape != asyn.shape:
        raise ValueError(
            f"sync and asyn must have the same shape; got {sync.shape} and {asyn.shape}"
        )

    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    plot_contour(
        sync,
        wavenumbers,
        wavenumbers_2=wavenumbers_2,
        ax=axes[0],
        title="Synchronous",
        cmap=cmap,
        n_levels=n_levels,
        descending=descending,
    )
    plot_contour(
        asyn,
        wavenumbers,
        wavenumbers_2=wavenumbers_2,
        ax=axes[1],
        title="Asynchronous",
        cmap=cmap,
        n_levels=n_levels,
        descending=descending,
        mask_diagonal=True,
    )
    if title is not None:
        fig.suptitle(title)
    return fig, axes


def _second_axis(wavenumbers_2, wavenumbers_x):
    """Resolve the ν₂ axis, honouring the deprecated `wavenumbers_x` alias"""
    if wavenumbers_x is None:
        return wavenumbers_2
    if wavenumbers_2 is not None:
        raise TypeError("pass wavenumbers_2 or the deprecate wavenumbers_x, not both")
    warnings.warn(
        "wavenumbers_x is deprecated; use wavenumbers_2. It is the ν₂ axis "
        "(matrix columns) and is drawn on the y-axis",
        DeprecationWarning,
        stacklevel=3,
    )
    return wavenumbers_x
