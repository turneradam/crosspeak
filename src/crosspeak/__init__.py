from crosspeak.io import read_series, read_spectrum, regrid_spectrum
from crosspeak.noda import asynchronous, hilbert_noda_matrix, synchronous
from crosspeak.plot import plot_contour
from crosspeak.preprocess import mean_center, reference_spectrum
from crosspeak.series import SpectralSeries

__version__ = "0.0.1"

__all__ = [
    "SpectralSeries",
    "asynchronous",
    "hilbert_noda_matrix",
    "mean_center",
    "plot_contour",
    "read_series",
    "read_spectrum",
    "reference_spectrum",
    "regrid_spectrum",
    "synchronous",
]
