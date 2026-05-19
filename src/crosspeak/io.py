from pathlib import Path

import numpy as np

from crosspeak.series import SpectralSeries


def read_spectrum(
    path,
    wavenumber_col=0,
    intensity_col=1,
    delimiter=",",
):
    """Read a spectrum from a file.

    Parameters
    ----------
    path : str or Path
        The path to the file containing the spectrum.
    wavenumber_col : int, optional
        The column index of the wavenumber values, by default 0.
    intensity_col : int, optional
        The column index of the intensity values, by default 1.
    delimiter : str, optional
        The delimiter used in the file, by default ",".

    Returns
    -------
    SpectralSeries
        A SpectralSeries object containing the wavenumber and intensity data.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.loadtxt(
        path,
        delimiter=delimiter,
        usecols=(wavenumber_col, intensity_col),
        ndmin=2,
    )

    if data.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in {path}, but got {data.shape[1]}")

    return data[:, 0], data[:, 1]


def read_series(
    files,
    name=None,
    wavenumber_col=0,
    intensity_col=1,
    delimiter=",",
):
    """Read a series of spectra from multiple files.

    Parameters
    ----------
    files : list of str or Path
        A list of paths to the files containing the spectra.
    name : str, optional
        The name of the series, by default None.
    wavenumber_col : int, optional
        The column index of the wavenumber values, by default 0.
    intensity_col : int, optional
        The column index of the intensity values, by default 1.
    delimiter : str, optional
        The delimiter used in the files, by default ",".

    Returns
    -------
    SpectralSeries
        A SpectralSeries object containing the spectra data.
    """
    if len(files) < 2:
        raise ValueError(f"need at least 2 files, got {len(files)}")

    perturbations = list(files.keys())
    paths = list(files.values())

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
                f"check your data or regrid the files to a common axis first"
            )
        intensities[i] = intens

    return SpectralSeries(
        wavenumbers=first_wn,
        perturbations=perturbations,
        intensities=intensities,
        name=name,
    )
