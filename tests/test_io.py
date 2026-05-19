import numpy as np
import pytest

from crosspeak import SpectralSeries, read_series, read_spectrum


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "spectrum.csv"
    wn = np.linspace(3700, 3100, 10)
    intens = np.sin(wn / 100)
    np.savetxt(path, np.column_stack([wn, intens]), delimiter=",")
    return path, wn, intens


def test_read_spectrum_basic(sample_csv):
    path, expected_wn, expected_intens = sample_csv
    wn, intens = read_spectrum(path)
    np.testing.assert_allclose(wn, expected_wn)
    np.testing.assert_allclose(intens, expected_intens)


def test_read_spectrum_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_spectrum(tmp_path / "nope.csv")


def test_read_spectrum_custom_columns(tmp_path):
    path = tmp_path / "multi.csv"
    n = 10
    wn = np.linspace(3700, 3100, n)
    intens = np.sin(wn / 100)
    extra = np.random.default_rng(0).normal(size=n)
    data = np.column_stack([np.arange(n), wn, intens, extra])
    np.savetxt(path, data, delimiter=",")

    wn_out, intens_out = read_spectrum(path, wavenumber_col=1, intensity_col=2)
    np.testing.assert_allclose(wn_out, wn)
    np.testing.assert_allclose(intens_out, intens)


@pytest.fixture
def three_csv_files(tmp_path):
    wn = np.linspace(3700, 3100, 20)
    files = {}
    for pert in [0, 5, 10]:
        p = tmp_path / f"sample_{pert}.csv"
        intens = np.sin(wn / 100) * (1 + pert / 10)
        np.savetxt(p, np.column_stack([wn, intens]), delimiter=",")
        files[pert] = p
    return files, wn


def test_read_series_basic(three_csv_files):
    files, expected_wn = three_csv_files
    s = read_series(files, name="test")
    assert isinstance(s, SpectralSeries)
    assert s.shape == (3, 20)
    assert s.name == "test"
    np.testing.assert_allclose(s.wavenumbers, expected_wn)
    np.testing.assert_array_equal(s.perturbations, [0, 5, 10])


def test_read_series_grid_mismatch_raises(tmp_path):
    wn_a = np.linspace(3700, 3100, 20)
    wn_b = np.linspace(3700, 3100, 25)

    pa = tmp_path / "a.csv"
    pb = tmp_path / "b.csv"
    np.savetxt(pa, np.column_stack([wn_a, np.zeros(20)]), delimiter=",")
    np.savetxt(pb, np.column_stack([wn_b, np.zeros(25)]), delimiter=",")

    with pytest.raises(ValueError, match="wavenumber grid"):
        read_series({0: pa, 1: pb})


def test_read_series_too_few_files():
    with pytest.raises(ValueError, match="at least 2"):
        read_series({0: "anywhere"})
