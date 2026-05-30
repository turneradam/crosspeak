import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Approximation of seaborn's vlag diverging palette (deep blue → near-white → brick red).
# 7 control points, interpolated to 256 levels by matplotlib.
# Refine with exact stops from seaborn source if perfect fidelity matters.
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


def plot_contour(
    matrix: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    wavenumbers_x: np.ndarray | None = None,
    ax=None,
    title: str | None = None,
    cmap: str = "vlag",
    n_levels: int = 50,
    descending: bool = True,
):
    """...

    Parameters
    ----------
    matrix
        2D correlation matrix.
    wavenumbers
        Wavenumber axis values for matrix axis 0 (rows, y-axis of plot).
        Length must equal matrix.shape[0].
    wavenumbers_x
        Optional. Wavenumber axis values for matrix axis 1 (cols, x-axis).
        Length must equal matrix.shape[1]. If None (default), `wavenumbers`
        is used for both axes — the standard homospectral case.
    ...
    """
    if wavenumbers_x is None:
        wavenumbers_x = wavenumbers

    matrix = np.asarray(matrix)
    wavenumbers = np.asarray(wavenumbers)

    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
    if matrix.shape[0] != wavenumbers.size:
        raise ValueError(
            f"matrix.shape[0] ({matrix.shape[0]}) must equal wavenumbers size ({wavenumbers.size})"
        )
    if matrix.shape[1] != wavenumbers_x.size:
        raise ValueError(
            f"matrix.shape[1] ({matrix.shape[1]}) must equal "
            f"wavenumbers_x size ({wavenumbers_x.size})"
        )
    if wavenumbers.ndim != 1:
        raise ValueError(f"wavenumbers must be 1D, got shape {wavenumbers.shape}")
    if matrix.shape[0] != wavenumbers.size:
        raise ValueError(
            f"matrix dimension {matrix.shape[0]} doesn't match "
            f"wavenumber axis length {wavenumbers.size}"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # Pin zero to the middle of the diverging colormap regardless of data asymmetry
    vmax = np.max(np.abs(matrix))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    cf = ax.contourf(
        wavenumbers_x,
        wavenumbers,
        matrix,
        levels=n_levels,
        cmap=cmap,
        norm=norm,
    )

    ax.contour(
        wavenumbers_x,
        wavenumbers,
        matrix,
        levels=10,
        colors="black",
        linewidths=0.4,
        alpha=0.6,
    )

    if descending:
        ax.set_xlim(wavenumbers.max(), wavenumbers.min())
        ax.set_ylim(wavenumbers.max(), wavenumbers.min())
    else:
        ax.set_xlim(wavenumbers.min(), wavenumbers.max())
        ax.set_ylim(wavenumbers.min(), wavenumbers.max())

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Wavenumber (cm⁻¹)")
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)

    plt.colorbar(cf, ax=ax, label="Correlation intensity")

    return ax
