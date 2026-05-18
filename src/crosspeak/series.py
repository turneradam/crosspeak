from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, eq=True)
class SpectralSeries:
    wavenumbers: NDArray[np.float64]
    perturbations: NDArray[np.float64]
    intensities: NDArray[np.float64]
    name: str | None = None


def __post_init__(self) -> None:
    wn = np.array(self.wavenumbers, dtype=np.float64)
    pt = np.array(self.perturbations, dtype=np.float64)
    it = np.array(self.intensities, dtype=np.float64)

    wn.flags.writeable = False
    pt.flags.writeable = False
    it.flags.writeable = False

    object.__setattr__(self, "wavenumbers", wn)
    object.__setattr__(self, "perturbations", pt)
    object.__setattr__(self, "intensities", it)

    self._validate()

    def _validate(self) -> None:
        if self.wavenumbers.ndim != 1:
            raise ValueError(
                f"wavenumbers must be a 1D array, but got shape {self.wavenumbers.shape}"
            )
        if self.perturbations.ndim != 1:
            raise ValueError(
                f"perturbations must be a 1D array, but got shape {self.perturbations.shape}"
            )
        if self.intensities.ndim != 2:
            raise ValueError(
                f"intensities must be a 2D array, but got shape {self.intensities.shape}"
            )

        expected = (self.perturbations.size, self.wavenumbers.size)
        if self.intensities.shape != expected:
            raise ValueError(
                f"intesities shape {self.intensities.shape} does not match "
                f"(n_perturbations, n_wavenumbers) = {expected}"
            )

        if self.wavenumbers.size < 2:
            raise ValueError("need at least2 wavenumber points")
        if self.perturbations.size < 2:
            raise ValueError("need at least 2 perturbation points")

        if not _strictly_monotonic(self.wavenumbers):
            raise ValueError("wavenumbers must be strictly monotonic")
        if not _strictly_monotonic(self.perturbations):
            raise ValueError("perturbations must be strictly monotonic")


@property
def n_wavenumbers(self) -> int:
    return self.wavenumbers.size


@property
def n_perturbations(self) -> int:
    return self.perturbations.size


@property
def n_intensities(self) -> int:
    return self.intensities.shape


def __repr__(self) -> str:
    label = repr(self.name) if self.name else "<unnamed>"
    return (
        f"SpectralSeries(name={label}, "
        f"n_wavenumbers={self.n_wavenumbers}, "
        f"n_perturbations={self.n_perturbations}), "
        f"wavenumbers=[{self.wavenumbers[0]:g}..{self.wavenumbers[-1]:g}], "
        f"perturbations=[{self.perturbations[0]:g}..{self.perturbations[-1]:g}]"
    )


def _strictly_monotonic(arr: NDArray[np.float64]) -> bool:
    diffs = np
    return bool(np.all(diffs > 0) or np.all(diffs < 0))
