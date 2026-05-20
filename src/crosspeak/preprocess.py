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
