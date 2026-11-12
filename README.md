# crosspeak

[![PyPI](https://img.shields.io/pypi/v/crosspeak)](https://pypi.org/project/crosspeak/)
[![CI](https://github.com/turneradam/crosspeak/actions/workflows/ci.yml/badge.svg)](https://github.com/turneradam/crosspeak/actions/workflows/ci.yml)
[![Docs](https://github.com/turneradam/crosspeak/actions/workflows/docs.yml/badge.svg)](https://turneradam.github.io/crosspeak/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21354308.svg)](https://doi.org/10.5281/zenodo.21354308)

Generalized two-dimensional correlation spectroscopy (2DCOS) in Python.
`crosspeak` takes a series of spectra recorded under a perturbation
(temperature, concentration, time) and computes Noda's synchronous and
asynchronous correlation maps, which show which bands change together and in
what order.

## Installation

```bash
pip install crosspeak
```

Requires Python 3.11 or later.

## Example

```python
import numpy as np
from crosspeak import SpectralSeries, synchronous, asynchronous, plot_sync_async

wn = np.arange(1000.0, 1400.0, 2.0)
t = np.linspace(0, 1, 11)
band = lambda centre: np.exp(-0.5 * ((wn - centre) / 15) ** 2)

# the 1300 band grows early in the series, the 1100 band late
spectra = np.array([(1 - np.exp(-x / 0.1)) * band(1300) + x**3 * band(1100) for x in t])
series = SpectralSeries(wavenumbers=wn, perturbations=t, intensities=spectra)

phi, psi = synchronous(series), asynchronous(series)
fig, axes = plot_sync_async(phi, psi, series.wavenumbers)
```

Matrices are indexed `[ν₁, ν₂]` and plotted with ν₁ on the x-axis. With that
convention, Noda's rule reads directly off the maps: if Φ(ν₁, ν₂) and
Ψ(ν₁, ν₂) have the same sign, the band at ν₁ changes before the band at ν₂.

## Features

- Synchronous and asynchronous maps, homo- and heterospectral
- Moving-window 2DCOS
- Autopeak detection
- Preprocessing: mean centring, cropping, regridding, Savitzky–Golay
  smoothing, area, vector and SNV normalization
- CSV readers for single spectra and series
- Contour plots with diverging colour maps

Full documentation and a worked tutorial: <https://turneradam.github.io/crosspeak/>

## Citing

If you use `crosspeak` in published work, please cite it using the DOI
[10.5281/zenodo.21354308](https://doi.org/10.5281/zenodo.21354308), which
always resolves to the latest release. Version-specific DOIs and a BibTeX entry
are on the Zenodo page and in [`CITATION.cff`](CITATION.cff).

## Contributing and support

Bug reports and feature requests go in the
[issue tracker](https://github.com/turneradam/crosspeak/issues). See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup and
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community standards.

## License

MIT. See [`LICENSE`](LICENSE).
