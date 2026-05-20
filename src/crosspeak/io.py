from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from crosspeak.series import SpectralSeries


def read_spectrum(
    path,
    wavenumber_col=0,
    intensity_col=1,
    delimiter=",",
):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")

    data = np.loadtxt(
        path,
        delimiter=delimiter,
        usecols=(wavenumber_col, intensity_col),
        ndmin=2,
    )

    if data.shape[1] != 2:
        raise ValueError(f"expected 2 columns from {path}, got shape {data.shape}")

    return data[:, 0], data[:, 1]


def regrid_spectrum(wn, intens, target_wn):
    wn = np.asarray(wn, dtype=np.float64)
    intens = np.asarray(intens, dtype=np.float64)
    target_wn = np.asarray(target_wn, dtype=np.float64)

    if wn.ndim != 1 or intens.ndim != 1:
        raise ValueError(f"wn and intens must be 1D, got shapes {wn.shape}, {intens.shape}")
    if wn.size != intens.size:
        raise ValueError(
            f"wn and intens must have the same length, got {wn.size} and {intens.size}"
        )
    if target_wn.ndim != 1:
        raise ValueError(f"target_wn must be 1D, got shape {target_wn.shape}")

    # CubicSpline needs strictly increasing x. FTIR files are often descending,
    # so sort to ascending and reorder the intensities to match.
    order = np.argsort(wn)
    wn = wn[order]
    intens = intens[order]

    if np.any(np.diff(wn) == 0):
        raise ValueError("wn contains duplicate values")

    src_min, src_max = wn[0], wn[-1]
    tgt_min, tgt_max = target_wn.min(), target_wn.max()
    if tgt_min < src_min or tgt_max > src_max:
        raise ValueError(
            f"target_wn range [{tgt_min:g}, {tgt_max:g}] extends outside "
            f"source range [{src_min:g}, {src_max:g}]; cannot extrapolate"
        )

    spline = CubicSpline(wn, intens)
    return spline(target_wn)


def read_series(
    files,
    name=None,
    *,
    target_grid=None,
    wavenumber_col=0,
    intensity_col=1,
    delimiter=",",
):
    if len(files) < 2:
        raise ValueError(f"need at least 2 files, got {len(files)}")

    perturbations = list(files.keys())
    paths = list(files.values())

    if target_grid is not None:
        target_grid = np.asarray(target_grid, dtype=np.float64)
        intensities = np.empty((len(files), target_grid.size))

        for i, p in enumerate(paths):
            wn, intens = read_spectrum(
                p,
                wavenumber_col=wavenumber_col,
                intensity_col=intensity_col,
                delimiter=delimiter,
            )
            intensities[i] = regrid_spectrum(wn, intens, target_grid)

        wavenumbers = target_grid
    else:
        first_wn, first_intens = read_spectrum(
            paths[0],
            wavenumber_col=wavenumber_col,
            intensity_col=intensity_col,
            delimiter=delimiter,
        )
        intensities = np.empty((len(files), first_wn.size))
        intensities[0] = first_intens

        for i, p in enumerate(paths[1:], start=1):
            wn, intens = read_spectrum(
                p,
                wavenumber_col=wavenumber_col,
                intensity_col=intensity_col,
                delimiter=delimiter,
            )
            if not np.array_equal(wn, first_wn):
                raise ValueError(
                    f"wavenumber grid in {p} differs from first file {paths[0]}; "
                    f"pass target_grid=... to regrid, or check your data"
                )
            intensities[i] = intens

        wavenumbers = first_wn

    return SpectralSeries(
        wavenumbers=wavenumbers,
        perturbations=perturbations,
        intensities=intensities,
        name=name,
    )
