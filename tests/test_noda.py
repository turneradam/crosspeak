import numpy as np
import pytest

from crosspeak import hilbert_noda_matrix


def test_shape():
    n = hilbert_noda_matrix(5)
    assert n.shape == (5, 5)


def test_zero_diagonal():
    n = hilbert_noda_matrix(7)
    np.testing.assert_array_equal(np.diag(n), np.zeros(7))


def test_antisymmetric():
    n = hilbert_noda_matrix(6)
    np.testing.assert_allclose(n, -n.T)


def test_specific_values_n3():
    n = hilbert_noda_matrix(3)
    expected = np.array(
        [
            [0.0, 1 / np.pi, 1 / (2 * np.pi)],
            [-1 / np.pi, 0.0, 1 / np.pi],
            [-1 / (2 * np.pi), -1 / np.pi, 0.0],
        ]
    )
    np.testing.assert_allclose(n, expected)


def test_off_diagonal_pattern():
    n = hilbert_noda_matrix(5)
    for k in range(1, 5):
        assert np.isclose(n[0, k], 1.0 / (np.pi * k))
        assert np.isclose(n[k, 0], -1.0 / (np.pi * k))


def test_minimum_size():
    n = hilbert_noda_matrix(2)
    assert n.shape == (2, 2)


def test_too_small_raises():
    with pytest.raises(ValueError, match="at least 2"):
        hilbert_noda_matrix(1)


def test_non_integer_raises():
    with pytest.raises(TypeError, match="must be an integer"):
        hilbert_noda_matrix(3.5)
