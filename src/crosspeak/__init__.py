from crosspeak.io import read_series, read_spectrum, regrid_spectrum
from crosspeak.noda import hilbert_noda_matrix
from crosspeak.preprocess import mean_center, reference_spectrum
from crosspeak.series import SpectralSeries

__version__ = "0.0.1"

__all__ = [
    "SpectralSeries",
    "hilbert_noda_matrix",
    "mean_center",
    "read_series",
    "read_spectrum",
    "reference_spectrum",
    "regrid_spectrum",
]
