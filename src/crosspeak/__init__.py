"""crosspeak — generalized 2D correlation spectroscopy for vibrational spectra."""

__version__ = "0.0.1"

from crosspeak.io import read_series, read_spectrum, regrid_spectrum
from crosspeak.noda import (
    AutopeakResult,
    asynchronous,
    find_autopeaks,
    hilbert_noda_matrix,
    synchronous,
)
from crosspeak.plot import plot_contour
from crosspeak.preprocess import mean_center, reference_spectrum
from crosspeak.series import SpectralSeries

__all__ = [
    "AutopeakResult",
    "SpectralSeries",
    "asynchronous",
    "find_autopeaks",
    "hilbert_noda_matrix",
    "mean_center",
    "plot_contour",
    "read_series",
    "read_spectrum",
    "reference_spectrum",
    "regrid_spectrum",
    "synchronous",
]
