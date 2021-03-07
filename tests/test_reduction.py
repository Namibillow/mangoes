import math

import logging
import pytest
import mangoes.reduction
import numpy as np
from scipy import sparse


logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


# ###########################################################################################
# ### Unit tests

@pytest.mark.unittest
def test_pca_null_matrix_dense():
    pca = mangoes.reduction.PCA(dimensions=5)
    result = pca(np.zeros((10, 20), dtype=float))

    expected = np.zeros((10, 5), dtype=int)

    np.testing.assert_array_equal(expected, result)


@pytest.mark.unittest
def test_svd_null_matrix_dense():
    matrix = np.zeros((10, 20), dtype=float)
    word_matrix, context_matrix = mangoes.reduction._svd(matrix, dimensions=5)

    assert word_matrix.shape == (10, 5)
    assert context_matrix.shape == (20, 5)

    expected_word_matrix = np.zeros((10, 5), dtype=int)
    np.testing.assert_array_equal(expected_word_matrix, word_matrix)


@pytest.mark.unittest
def test_svd_null_matrix_sparse():
    matrix = sparse.csr_matrix((10, 20), dtype=float)
    word_matrix, context_matrix = mangoes.reduction._svd(matrix, dimensions=5)

    assert word_matrix.shape == (10, 5)
    assert context_matrix.shape == (20, 5)

    expected_word_matrix = np.zeros((10, 5), dtype=int)
    np.testing.assert_array_equal(expected_word_matrix, word_matrix)


@pytest.mark.unittest
def test_svd_weight():
    import math
    matrix = np.array([1, 0, 0, 0, 2,
                       0, 0, 3, 0, 0,
                       0, 0, 0, 0, 0,
                       0, 4, 0, 0, 0], dtype=float).reshape((4, 5))

    Ud = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0],
                   [0, 0, 1]])
    Sigmad = np.array([[math.sqrt(5), 0, 0],
                       [0, 3, 0],
                       [0, 0, 4]])

    Vd = np.array([[math.sqrt(5) / 5, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 0, 0],
                   [2 * math.sqrt(5) / 5, 0, 0]])

    # weight = 1 (classic SVD)
    word_matrix, context_matrix = mangoes.reduction._svd(matrix, dimensions=3, weight=1, symmetric=False)

    assert (4, 3) == word_matrix.shape
    assert (5, 3) == context_matrix.shape
    np.testing.assert_array_almost_equal(Ud.dot(Sigmad), np.abs(word_matrix))
    np.testing.assert_array_almost_equal(Vd, np.abs(context_matrix))

    # weight = 0
    word_matrix, context_matrix = mangoes.reduction._svd(matrix, dimensions=3, weight=0, symmetric=True)

    assert (4, 3) == word_matrix.shape
    assert (5, 3) == context_matrix.shape

    np.testing.assert_array_almost_equal(Ud, np.abs(word_matrix))
    np.testing.assert_array_almost_equal(Vd, np.abs(context_matrix))

    # weight = 0.5
    word_matrix, context_matrix = mangoes.reduction._svd(matrix, dimensions=3, weight=0.5)

    assert (4, 3) == word_matrix.shape
    assert (5, 3) == context_matrix.shape
    np.testing.assert_array_almost_equal(Ud.dot(np.sqrt(Sigmad)), np.abs(word_matrix))
    np.testing.assert_array_almost_equal(Vd.dot(np.sqrt(Sigmad)), np.abs(context_matrix))


########################
# Exceptions
def test_exception_svd_incompatible_dimension():
    matrix = np.array([1, 0, 0, 0, 2,
                       0, 0, 3, 0, 0,
                       0, 0, 0, 0, 0,
                       0, 4, 0, 0, 0], dtype=float).reshape((4, 5))
    with pytest.raises(mangoes.utils.exceptions.IncompatibleValue):
        mangoes.reduction.SVD(dimensions=5)(matrix)

    sp_matrix = sparse.csr_matrix(matrix)
    with pytest.raises(mangoes.utils.exceptions.IncompatibleValue):
        mangoes.reduction.SVD(dimensions=5)(sp_matrix)
