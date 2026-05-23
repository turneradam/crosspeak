import numpy as np
import pytest

from crosspeak import SpectralSeries, read_series, read_spectrum, regrid_spectrum


def test_regrid_basic():
    src_wn = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    src_intens = src_wn**2
    target_wn = np.array([1.5, 2.5, 3.5])

    out = regrid_spectrum(src_wn, src_intens, target_wn)

    np.testing.assert_allclose(out, target_wn**2, atol=1e-10)


def test_regrid_identity():
    wn = np.linspace(3700, 3100, 50)
    intens = np.sin(wn / 100)

    out = regrid_spectrum(wn, intens, wn)

    np.testing.assert_allclose(out, intens, atol=1e-10)


def test_regrid_descending_source():
    wn_desc = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    intens = wn_desc**2
    target_wn = np.array([1.5, 2.5, 3.5])

    out = regrid_spectrum(wn_desc, intens, target_wn)

    np.testing.assert_allclose(out, target_wn**2, atol=1e-10)


def test_regrid_out_of_range_raises():
    wn = np.linspace(3700, 3100, 50)
    intens = np.sin(wn / 100)
    target_wn = np.linspace(4000, 3000, 30)

    with pytest.raises(ValueError, match="extends outside"):
        regrid_spectrum(wn, intens, target_wn)


def test_regrid_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        regrid_spectrum([1, 2, 3], [1, 2, 3, 4], [1.5])


def test_regrid_duplicate_wavenumbers_raises():
    with pytest.raises(ValueError, match="duplicate"):
        regrid_spectrum([1, 2, 2, 3], [1, 4, 4, 9], [1.5])


def test_read_series_with_target_grid(tmp_path):
    wn_a = np.linspace(3700, 3100, 20)
    wn_b = np.linspace(3699, 3101, 22)  # different grid

    pa = tmp_path / "a.csv"
    pb = tmp_path / "b.csv"
    np.savetxt(pa, np.column_stack([wn_a, np.sin(wn_a / 100)]), delimiter=",")
    np.savetxt(pb, np.column_stack([wn_b, np.sin(wn_b / 100) * 1.5]), delimiter=",")

    target = np.linspace(3699, 3101, 30)
    s = read_series({0: pa, 1: pb}, target_grid=target)

    assert s.shape == (2, 30)
    np.testing.assert_allclose(s.wavenumbers, target)


def test_read_series_target_grid_out_of_range(tmp_path):
    wn = np.linspace(3700, 3100, 20)
    pa = tmp_path / "a.csv"
    pb = tmp_path / "b.csv"
    np.savetxt(pa, np.column_stack([wn, np.zeros(20)]), delimiter=",")
    np.savetxt(pb, np.column_stack([wn, np.zeros(20)]), delimiter=",")

    target = np.linspace(4000, 3000, 30)

    with pytest.raises(ValueError, match="extends outside"):
        read_series({0: pa, 1: pb}, target_grid=target)


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


def test_read_spectrum_with_header_row(tmp_path):
    path = tmp_path / "with_header.csv"
    wn = np.linspace(3700, 3100, 10)
    intens = np.sin(wn / 100)
    with open(path, "w") as f:
        f.write("Wavenumber,Absorbance\n")
        for w, a in zip(wn, intens, strict=True):
            f.write(f"{w},{a}\n")

    wn_out, intens_out = read_spectrum(path, skiprows=1)
    np.testing.assert_allclose(wn_out, wn)
    np.testing.assert_allclose(intens_out, intens)


def test_read_spectrum_header_without_skiprows_gives_useful_error(tmp_path):
    path = tmp_path / "with_header.csv"
    with open(path, "w") as f:
        f.write("Wavenumber,Absorbance\n1.0,0.1\n2.0,0.2\n")

    with pytest.raises(ValueError, match="skiprows"):
        read_spectrum(path)
