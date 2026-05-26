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
