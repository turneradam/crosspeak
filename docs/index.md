# crosspeak

Generalized two-dimensional correlation spectroscopy (2DCOS) for vibrational
spectra. Synchronous and asynchronous maps, heterospectral and moving-window
correlation, preprocessing, and plotting — installable, tested, and validated
against an independent implementation.

## Install

```bash
pip install crosspeak
```

## Quickstart

```python
from crosspeak import read_series, savgol_smooth, crop_region, synchronous, asynchronous

# one file per water content, keyed by the perturbation value
series = read_series({0: "0w.csv", 5: "5w.csv", 10: "10w.csv", 20: "20w.csv"})

oh = crop_region(savgol_smooth(series), 3000, 3700)
phi = synchronous(oh)   # which bands move together
psi = asynchronous(oh)  # and in what order
```

See the tutorials for worked examples and the API reference for the full surface.