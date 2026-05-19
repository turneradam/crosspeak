# Roadmap

## v0.0.x — Foundation
Package scaffold; no public API. Verifies build, test, lint, and CI.

## v0.1 — Vocabulary
`SpectralSeries` data structure. CSV reader. CITATION.cff at repo root.
Zenodo–GitHub integration enabled. First PyPI release. v0.1 mints first
Zenodo DOI.

## v0.2 — Core correlation
Synchronous, asynchronous (Hilbert–Noda), autopeak detection on the diagonal.

## v0.3 — Preprocessing
Savitzky–Golay smoothing, area normalization, mean-centering, common-grid interpolation, region selection.

## v0.4 — Plotting
Contour maps with publication defaults, overlays, slice extraction.
Hard-coded seaborn-style palettes (rocket, mako, flare, crest, vlag,
icefire) under `crosspeak.plot.palettes`. No seaborn runtime dependency.

## v0.5 — Heterospectral + moving-window
Maps between distinct spectral regions; MW2D under slowly varying perturbations.

## v0.6 — I/O expansion
JCAMP-DX reader. OPUS reader deferred.

## v0.7 — Validation
Reproduce a published 2DCOS result. Non-negotiable before JOSS submission.

## v0.8 — Tutorials and docs
Two Jupyter notebooks (one synthetic, one with real DES-water data). Sphinx or MkDocs API reference.

## v0.9 
`paper.md`, `paper.bib`, README badges, polish. (CITATION.cff and
Zenodo already minted at v0.1.)

## v1.0 
Tagged release archived on Zenodo; paper submitted.

## Post-v1.0
Stretch features: PCMW2D, codistribution, sample-sample, higher-order 2DCOS.