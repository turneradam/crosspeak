from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from crosspeak.series import SpectralSeries


def read_spectrum(path, wavenumber_col=0, intensity_col=1, delimiter=",", skiprows=0):
    """Read a single spectrum from a delimited text file.

    Pulls two columns (wavenumber and intensity) out of a CSV or similar.
    Everything else in the file is ignored, so an export carrying extra
    metadata columns is fine as long as you point at the right two.

    Parameters
    ----------
    path
        File to read.
    wavenumber_col, intensity_col
        Zero-based column indices; default to the first two columns.
    delimiter
        Column separator, comma by default. Use "\\t" for tab-separated exports.
    skiprows
        Header lines to skip. If the read fails complaining that it can't turn
        a string into a float, it's almost always a header, set this to 1.

    Returns
    -------
    tuple of np.ndarray
        `(wavenumbers, intensities)`, each 1-D and in file order.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")

    try:
        data = np.loadtxt(
            path,
            delimiter=delimiter,
            usecols=(wavenumber_col, intensity_col),
            ndmin=2,
            skiprows=skiprows,
        )
    except ValueError as e:
        raise ValueError(
            f"failed to read {path}: {e}. If the file has a header row, pass skiprows=1."
        ) from e

    if data.shape[1] != 2:
        raise ValueError(f"expected 2 columns from {path}, got shape {data.shape}")

    return data[:, 0], data[:, 1]


def regrid_spectrum(wn, intens, target_wn):
    """Resample a spectrum onto a new wavenumber grid by cubic spline.

    Needed whenever the spectra in a series don't share one axis, different
    instruments, or the same instrument drifting point to point, before they
    can be stacked into a `SpectralSeries`. A natural cubic spline is fit to the
    data and evaluated at `target_wn`.

    Descending wavenumber input is handled; the data
    is sorted ascending internally, so order doesn't matter. Extrapolation is
    refused `target_wn` has to sit inside the source range, or you'd be
    inventing intensity past the measured edges.

    Parameters
    ----------
    wn, intens
        Source wavenumbers and intensities, same length.
    target_wn
        Wavenumbers to resample onto. Must lie within the span of `wn`.

    Returns
    -------
    np.ndarray
        Intensities on `target_wn`.

    Raises
    ------
    ValueError
        If lengths disagree, `wn` has duplicates, or `target_wn` runs past the
        source range.
    """
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
    skiprows=0,
):
    """Read several spectra into a single `SpectralSeries`.

    `files` maps each perturbation value to the file holding that spectrum, e.g.
    `{0: "0w.csv", 5: "5w.csv", 10: "10w.csv"}`. The keys become the
    perturbation axis; the files are read in that order and stacked into rows.

    By default every file must already sit on an identical wavenumber grid, and
    a mismatch is raised rather than silently interpolated away. If the grids
    genuinely differ, pass `target_grid` and each spectrum is splined onto it on
    the way in (see `regrid_spectrum`).

    Parameters
    ----------
    files
        Mapping of perturbation value to file path; at least two entries.
    name
        Optional label carried on the resulting series.
    target_grid
        Common wavenumber axis to regrid onto. If omitted, the source grids
        must already match exactly.
    wavenumber_col, intensity_col, delimiter, skiprows
        Passed through to `read_spectrum` for every file.

    Returns
    -------
    SpectralSeries
        Perturbations from the keys, one intensity row per file.

    Raises
    ------
    ValueError
        With fewer than two files, or when grids differ and no `target_grid`
        was given.
    """
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
                skiprows=skiprows,
            )
            intensities[i] = regrid_spectrum(wn, intens, target_grid)

        wavenumbers = target_grid
    else:
        first_wn, first_intens = read_spectrum(
            paths[0],
            wavenumber_col=wavenumber_col,
            intensity_col=intensity_col,
            delimiter=delimiter,
            skiprows=skiprows,
        )
        intensities = np.empty((len(files), first_wn.size))
        intensities[0] = first_intens

        for i, p in enumerate(paths[1:], start=1):
            wn, intens = read_spectrum(
                p,
                wavenumber_col=wavenumber_col,
                intensity_col=intensity_col,
                delimiter=delimiter,
                skiprows=skiprows,
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
