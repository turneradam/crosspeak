import numpy as np
from scipy.signal import savgol_filter

from crosspeak.series import SpectralSeries


def reference_spectrum(series):
    return series.intensities.mean(axis=0)


def mean_center(series):
    reference = reference_spectrum(series)
    dynamic = series.intensities - reference
    return SpectralSeries(
        wavenumbers=series.wavenumbers,
        perturbations=series.perturbations,
        intensities=dynamic,
        name=series.name,
    )


def crop_region(series: SpectralSeries, low: float, high: float) -> SpectralSeries:
    """Return a new SpectralSeries restricted to a wavenumber range
    Both bounds are inclusive. Does not modify original series.
    Wavenumber axis direction is maintained
    """

    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")

    wn = series.wavenumbers
    mask = (wn >= low) & (wn <= high)

    if not mask.any():
        raise ValueError(
            f"requested range [{low}, {high}] has no overlap "
            f"with data range [{wn.min()}, {wn.max()}]"
        )

    return SpectralSeries(
        wavenumbers=wn[mask],
        perturbations=series.perturbations,
        intensities=series.intensities[:, mask],
        name=series.name,
    )


def savgol_smooth(
    series: SpectralSeries,
    window_length: int = 13,
    polyorder: int = 3,
    **kwargs,
) -> SpectralSeries:
    """Apply SavitzkyGolay smoothing along the wavenumber axis.

    Each spectrum (row) is smoothed independently. The original series is
    not modified.

    Parameters
    ----------
    series
        Input SpectralSeries.
    window_length
        Length of the filter window. Must be odd, greater than `polyorder`,
        and no larger than the number of wavenumber points. Default 13.
    polyorder
        Order of the polynomial fit. Must be less than `window_length`.
        Default 3.
    **kwargs
        Additional keyword arguments passed through to
        `scipy.signal.savgol_filter` (e.g. `deriv`, `delta`, `mode`, `cval`).
        Do not pass `axis` — it is fixed to the wavenumber axis.

    Returns
    -------
    SpectralSeries
        New SpectralSeries with smoothed intensities. Wavenumbers,
        perturbations, and name are preserved.
    """
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")

    smoothed = savgol_filter(
        series.intensities,
        window_length=window_length,
        polyorder=polyorder,
        axis=-1,
        **kwargs,
    )

    return SpectralSeries(
        wavenumbers=series.wavenumbers,
        perturbations=series.perturbations,
        intensities=smoothed,
        name=series.name,
    )


def area_normalize(
    series: SpectralSeries,
    target_area: float = 1.0,
    reference_region: tuple[float, float] | None = None,
) -> SpectralSeries:
    """Normalize each spectrum to a target area under the curve.

    Parameters
    ----------
    series
        Input SpectralSeries.
    target_area
        Desired area under the curve for each spectrum after normalization.
        Default is 1.0.
    reference_region
        Optional tuple specifying a wavenumber range (low, high) to use for
        calculating the area. If None, the entire wavenumber range is used.

    Returns
    -------
    SpectralSeries
        New SpectralSeries with normalized intensities. Wavenumbers,
        perturbations, and name are preserved.

    Raises
    ------
    ValueError
        If any row's integrated area is zero, or if `reference_region` is
        invalid (inverted bounds or no overlap with data — delegated to
        `crop_region`).
    """
    if reference_region is not None:
        ref = crop_region(series, *reference_region)
        wn_ref = ref.wavenumbers
        intens_ref = ref.intensities
    else:
        wn_ref = series.wavenumbers
        intens_ref = series.intensities

    areas = np.abs(np.trapezoid(intens_ref, x=wn_ref, axis=-1))

    if (areas == 0).any():
        zero_rows = np.where(areas == 0)[0].tolist()
        raise ValueError(f"zero integrated area on perturbation rows {zero_rows}; cannot normalize")

    scale = target_area / areas
    normalized = series.intensities * scale[:, None]

    return SpectralSeries(
        wavenumbers=series.wavenumbers,
        perturbations=series.perturbations,
        intensities=normalized,
        name=series.name,
    )
