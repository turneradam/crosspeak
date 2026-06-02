"""crosspeak — generalized 2D correlation spectroscopy for vibrational spectra."""

from importlib.metadata import version

__version__ = version("crosspeak")

from crosspeak.io import read_series, read_spectrum, regrid_spectrum
from crosspeak.noda import (
    AutopeakResult,
    asynchronous,
    asynchronous_hetero,
    find_autopeaks,
    hilbert_noda_matrix,
    synchronous,
    synchronous_hetero,
)
from crosspeak.plot import plot_contour, plot_sync_async
from crosspeak.preprocess import (
    area_normalize,
    crop_region,
    mean_center,
    reference_spectrum,
    savgol_smooth,
)
from crosspeak.series import SpectralSeries

__all__ = [
    "AutopeakResult",
    "SpectralSeries",
    "area_normalize",
    "asynchronous",
    "asynchronous_hetero",
    "crop_region",
    "find_autopeaks",
    "hilbert_noda_matrix",
    "mean_center",
    "plot_contour",
    "plot_sync_async",
    "read_series",
    "read_spectrum",
    "reference_spectrum",
    "regrid_spectrum",
    "savgol_smooth",
    "synchronous",
    "synchronous_hetero",
]
