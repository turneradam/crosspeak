# Changelog

## 1.1.0 — 2026-10-10

### Fixed
- Asynchronous and heterospectral contour plots were drawn transposed: the point
  read as (ν₁, ν₂) showed Ψ(ν₂, ν₁) = −Ψ(ν₁, ν₂), inverting sequential-order
  readings. Plots now draw ν₁ on the x-axis and label both axes. Computed
  matrices are unchanged.

### Added
- `moving_window` and `MovingWindowResult`: moving-window 2DCOS.
- `wavenumbers_2` keyword on `plot_contour` and `plot_sync_async`.

### Deprecated
- `wavenumbers_x`; use `wavenumbers_2`.

## 1.0.0 — 2026-07-14
- First public release.
