# -*- coding: utf-8 -*-

import numpy as np
import pytest
import scipy
import scipy.sparse

import mangoes.utils.arrays
import mangoes.utils.exceptions

# ###########################################################################################
# ### Unit tests


# Normalization

l1_data = [
    # origin                l1_normalized
    ([[-3, 4], [2, -2]], [[-3 / 7, 4 / 7], [2 / 4, -2 / 4]]),
    ([-3, 4], [-3 / 7, 4 / 7]),
    ([[-3], [4]], [[-1], [1]]),
    ([[-3, 4], [0, 0]], [[-3 / 7, 4 / 7], [0, 0]]),
    ([0, 0], [0, 0])
]

l2_data = [
    # origin                l2_normalized
    ([[3, 4], [2, 2]], [[3 / 5, 4 / 5], [2 / np.sqrt(8), 2 / np.sqrt(8)]]),
    ([3, 4], [3 / 5, 4 / 5]),
    ([[3], [4]], [[1], [1]]),
    ([[3, 4], [0, 0]], [[3 / 5, 4 / 5], [0, 0]]),
    ([0, 0], [0, 0])
]


@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_indexing_returns_a_mangoes_matrix(matrix_type):
    matrix = matrix_type(l1_data[0][0])
    assert isinstance(matrix[0], mangoes.utils.arrays.Matrix)


@pytest.mark.unittest
@pytest.mark.parametrize("origin, l1_normalized", l1_data)
def test_dense_l1_normalize_rows(origin, l1_normalized):
    matrix = mangoes.utils.arrays.NumpyMatrix(np.array(origin))
    expected = mangoes.utils.arrays.NumpyMatrix(l1_normalized)
    assert np.allclose(expected, matrix.normalize(norm="l1", axis=1))


@pytest.mark.unittest
@pytest.mark.parametrize("origin, l1_normalized", l1_data)
def test_sparse_l1_normalize_rows(origin, l1_normalized):
    matrix = mangoes.utils.arrays.csrSparseMatrix(scipy.sparse.csr_matrix(origin))
    expected = mangoes.utils.arrays.csrSparseMatrix(l1_normalized)
    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected, matrix.normalize(norm="l1", axis=1))


@pytest.mark.unittest
@pytest.mark.parametrize("origin, l2_normalized", l2_data)
def test_dense_l2_normalize_rows(origin, l2_normalized):
    matrix = mangoes.utils.arrays.NumpyMatrix(np.array(origin))
    expected = mangoes.utils.arrays.NumpyMatrix(l2_normalized)
    assert np.allclose(expected, mangoes.utils.arrays.normalize(matrix, norm="l2", axis=1))


@pytest.mark.unittest
@pytest.mark.parametrize("origin, l2_normalized", l2_data)
def test_sparse_l2_normalize_rows(origin, l2_normalized):
    matrix = mangoes.utils.arrays.csrSparseMatrix(scipy.sparse.csr_matrix(origin))
    expected = mangoes.utils.arrays.csrSparseMatrix(l2_normalized)
    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected, matrix.normalize(norm="l2", axis=1))


@pytest.mark.unittest
@pytest.mark.skip(reason="Inplace doesn't work yet")
def test_normalize_inplace_1d():
    normalize = mangoes.utils.arrays.normalize
    matrix1 = mangoes.utils.arrays.NumpyMatrix(np.array([3.0, 4.0]))

    expected_result1 = mangoes.utils.arrays.NumpyMatrix(np.array([3 / 5, 4 / 5]).T)

    actual_result1 = normalize(matrix1, norm="l2", axis=0, inplace=True)

    assert np.allclose(expected_result1, actual_result1)
    assert matrix1 is not actual_result1


@pytest.mark.unittest
@pytest.mark.skip(reason="Inplace doesn't work yet")
def test_normalize_inplace_np():
    normalize = mangoes.utils.arrays.normalize
    matrix1 = mangoes.utils.arrays.NumpyMatrix(np.array([[3.0, 4.0], [2.0, 2.0]]))
    matrix2 = mangoes.utils.arrays.NumpyMatrix(np.array([[3.0, 2.0], [4.0, 2.0]]))

    expected_result1 = mangoes.utils.arrays.NumpyMatrix(np.array([[3 / np.sqrt(13), 2 / np.sqrt(13)],
                                                                  [4 / np.sqrt(20), 2 / np.sqrt(20)]]).T)
    expected_result2 = mangoes.utils.arrays.NumpyMatrix(np.array([[3 / np.sqrt(13), 2 / np.sqrt(13)],
                                                                  [4 / np.sqrt(20), 2 / np.sqrt(20)]]))

    actual_result1 = normalize(matrix1, norm="l2", axis=0, inplace=True)
    actual_result2 = normalize(matrix2, norm="l2", axis=1, inplace=True)

    assert np.allclose(expected_result1, actual_result1)
    assert matrix1 is not actual_result1
    assert np.allclose(expected_result2, actual_result2)
    assert matrix2 is actual_result2


@pytest.mark.unittest
def test_normalize_inplace_sparse():
    normalize = mangoes.utils.arrays.normalize
    matrix1 = mangoes.utils.arrays.csrSparseMatrix(np.array([[3.0, 4.0], [2.0, 2.0]]))
    matrix2 = mangoes.utils.arrays.csrSparseMatrix(np.array([[3.0, 2.0], [4.0, 2.0]]))

    expected_result1 = mangoes.utils.arrays.csrSparseMatrix(np.array([[3 / np.sqrt(13), 2 / np.sqrt(13)],
                                                                      [4 / np.sqrt(20), 2 / np.sqrt(20)]]).T)
    expected_result2 = mangoes.utils.arrays.csrSparseMatrix(np.array([[3 / np.sqrt(13), 2 / np.sqrt(13)],
                                                                      [4 / np.sqrt(20), 2 / np.sqrt(20)]]))

    actual_result1 = normalize(matrix1, norm="l2", axis=0, inplace=True)
    actual_result2 = normalize(matrix2, norm="l2", axis=1, inplace=True)

    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result1, actual_result1)
    assert matrix1 is not actual_result1
    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result2, actual_result2)
    assert matrix2 is actual_result2


@pytest.mark.unittest
@pytest.mark.skip(reason="Wait implementing column normalization")
def test_l1_normalize_colums():
    normalize = mangoes.utils.arrays.normalize
    matrix1 = np.array([[-3, 4], [2, -2]])
    matrix2 = np.array([-3, 4])
    matrix25 = np.array([[-3, 4]])
    matrix3 = np.array([[-3, 0], [4, 0]])
    matrix4 = np.array([0, 0])

    expected_result1 = np.transpose(np.array([[-3 / 5, 2 / 5],
                                              [4 / 6, -2 / 6]]))
    expected_result2 = np.array([-3 / 7, 4 / 7])
    expected_result25 = np.array([[-1, 1]])
    expected_result3 = np.transpose(np.array([[-3 / 7, 4 / 7], [0, 0]]))
    expected_result4 = np.array([0, 0])
    actual_result1 = normalize(matrix1, norm="l1", axis=0)
    actual_result2 = normalize(matrix2, norm="l1", axis=0)
    actual_result25 = normalize(matrix25, norm="l1", axis=0)
    actual_result3 = normalize(matrix3, norm="l1", axis=0)
    actual_result4 = normalize(matrix4, norm="l1", axis=0)

    assert np.allclose(expected_result1, actual_result1)
    assert np.allclose(expected_result2, actual_result2)
    assert np.allclose(expected_result25, actual_result25)
    assert np.allclose(expected_result3, actual_result3)
    assert np.allclose(expected_result4, actual_result4)


@pytest.mark.unittest
@pytest.mark.skip(reason="Wait implementing column normalization")
def test_sparse_l1_normalize_colums():
    normalize = mangoes.utils.arrays.normalize
    matrix1 = scipy.sparse.csc_matrix([[-3, 4], [2, -2]])
    matrix2 = scipy.sparse.csc_matrix([[-3], [4]])
    matrix25 = scipy.sparse.csc_matrix([[-3, 4]])
    matrix3 = scipy.sparse.csc_matrix([[-3, 0], [4, 0]])
    matrix4 = scipy.sparse.csc_matrix([0, 0]).T

    expected_result1 = scipy.sparse.csr_matrix([[-3 / 5, 2 / 5], [4 / 6, -2 / 6]]).T
    expected_result2 = scipy.sparse.csr_matrix([[-3 / 7, 4 / 7]]).T
    expected_result25 = scipy.sparse.csc_matrix([[-1, 1]])
    expected_result3 = scipy.sparse.csr_matrix([[-3 / 7, 4 / 7], [0, 0]]).T
    expected_result4 = scipy.sparse.csr_matrix([0, 0]).T
    actual_result1 = normalize(matrix1, norm="l1", axis=0)
    actual_result2 = normalize(matrix2, norm="l1", axis=0)
    actual_result25 = normalize(matrix25, norm="l1", axis=0)
    actual_result3 = normalize(matrix3, norm="l1", axis=0)
    actual_result4 = normalize(matrix4, norm="l1", axis=0)

    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result1, actual_result1)
    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result2, actual_result2)
    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result25, actual_result25)
    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result3, actual_result3)
    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result4, actual_result4)


@pytest.mark.unittest
@pytest.mark.skip(reason="Wait implementing column normalization")
def test_l2_normalize_colums():
    normalize = mangoes.utils.arrays.normalize
    matrix1 = np.array([[3, 4], [2, 2]])
    matrix2 = np.array([3, 4])
    matrix25 = np.array([[3, 4]])
    matrix3 = np.array([[3, 0], [4, 0]])
    matrix4 = np.array([0, 0])

    expected_result1 = np.transpose(np.array([[3 / np.sqrt(13), 2 / np.sqrt(13)],
                                              [4 / np.sqrt(20), 2 / np.sqrt(20)]]))
    expected_result2 = np.array([3 / 5, 4 / 5])
    expected_result25 = np.array([[1, 1]])
    expected_result3 = np.transpose(np.array([[3 / 5, 4 / 5], [0, 0]]))
    expected_result4 = np.array([0, 0])

    actual_result1 = normalize(matrix1, norm="l2", axis=0)
    actual_result2 = normalize(matrix2, norm="l2", axis=0)
    actual_result25 = normalize(matrix25, norm="l2", axis=0)
    actual_result3 = normalize(matrix3, norm="l2", axis=0)
    actual_result4 = normalize(matrix4, norm="l2", axis=0)

    assert np.allclose(expected_result1, actual_result1)
    assert np.allclose(expected_result2, actual_result2)
    assert np.allclose(expected_result25, actual_result25)
    assert np.allclose(expected_result3, actual_result3)
    assert np.allclose(expected_result4, actual_result4)


@pytest.mark.unittest
@pytest.mark.skip(reason="Wait implementing column normalization")
def test_sparse_l2_normalize_colums():
    normalize = mangoes.utils.arrays.normalize
    matrix1 = scipy.sparse.csc_matrix([[3, 4], [2, 2]])
    matrix2 = scipy.sparse.csc_matrix([[3], [4]])
    matrix25 = scipy.sparse.csc_matrix([[3, 4]])
    matrix3 = scipy.sparse.csc_matrix([[3, 0], [4, 0]])
    matrix4 = scipy.sparse.csc_matrix([0, 0])

    expected_result1 = scipy.sparse.csr_matrix([[3 / np.sqrt(13), 2 / np.sqrt(13)],
                                                [4 / np.sqrt(20), 2 / np.sqrt(20)]]).T
    expected_result2 = scipy.sparse.csc_matrix([[3 / 5], [4 / 5]])
    expected_result25 = scipy.sparse.csc_matrix([[1, 1]])
    expected_result3 = scipy.sparse.csr_matrix([[3 / 5, 4 / 5], [0, 0]]).T
    expected_result4 = scipy.sparse.csc_matrix([0, 0])
    actual_result1 = normalize(matrix1, norm="l2", axis=0)
    actual_result2 = normalize(matrix2, norm="l2", axis=0)
    actual_result25 = normalize(matrix25, norm="l2", axis=0)
    actual_result3 = normalize(matrix3, norm="l2", axis=0)
    actual_result4 = normalize(matrix4, norm="l2", axis=0)

    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result1, actual_result1)
    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result2, actual_result2)
    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result25, actual_result25)
    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result3, actual_result3)
    assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result4, actual_result4)


# Center

@pytest.mark.unittest
def test_center_np():
    matrix1 = mangoes.utils.arrays.NumpyMatrix(np.array([[3.0, 4.0], [2.0, 2.0]]))
    matrix2 = mangoes.utils.arrays.NumpyMatrix(np.array([[3.0, 4.0], [2.0, 2.0]]))

    expected_result1 = mangoes.utils.arrays.NumpyMatrix(np.array([[0.5, 1.0], [-0.5, -1.0]]))
    expected_result2 = mangoes.utils.arrays.NumpyMatrix(np.array([[-0.5, 0.5], [0.0, 0.0]]))

    actual_result1 = mangoes.utils.arrays.center(matrix1, axis=0)
    actual_result2 = mangoes.utils.arrays.center(matrix2, axis=1)

    assert np.allclose(expected_result1, actual_result1)
    # assert matrix1 is actual_result1
    assert np.allclose(expected_result2, actual_result2)
    # assert matrix2 is actual_result2


@pytest.mark.unittest
def test_center_sparse():
    matrix1 = mangoes.utils.arrays.csrSparseMatrix(scipy.sparse.csc_matrix([[3.0, 4.0], [2.0, 2.0]]))
    matrix2 = mangoes.utils.arrays.csrSparseMatrix(scipy.sparse.csr_matrix([[3.0, 4.0], [2.0, 2.0]]))

    expected_result1 = mangoes.utils.arrays.NumpyMatrix(np.array([[0.5, 1.0], [-0.5, -1.0]]))
    expected_result2 = mangoes.utils.arrays.NumpyMatrix(np.array([[-0.5, 0.5], [0.0, 0.0]]))

    actual_result1 = mangoes.utils.arrays.center(matrix1, axis=0)  # , inplace=True)
    actual_result2 = mangoes.utils.arrays.center(matrix2, axis=1)  # , inplace=True)

    assert np.allclose(expected_result1, actual_result1)
    assert np.allclose(expected_result2, actual_result2)


# Square root

@pytest.mark.unittest
@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_sqrt(matrix_type):
    origin = matrix_type([[9.0, 4.0], [4.0, 1.0]])
    expected = matrix_type([[3.0, 2.0], [2.0, 1.0]])

    result1 = mangoes.utils.arrays.sqrt(origin, inplace=False)

    assert result1 is not origin
    try:
        assert np.allclose(expected, result1)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected, result1)

    result2 = mangoes.utils.arrays.sqrt(origin, inplace=True)

    assert result2 is origin
    try:
        assert np.allclose(expected, result2)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected, result2)


# All positives

@pytest.mark.unittest
@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_all_positive(matrix_type):
    matrix = matrix_type(np.matrix([[0, 0, 1],
                                    [0, 0, 2],
                                    [0, 5, 1]]))

    assert matrix.all_positive()

    matrix = matrix_type(np.matrix([[0, 0, 1],
                                    [0, 0, 2],
                                    [0, -5, 1]]))

    assert not matrix.all_positive()


# test different ways to combine matrices
@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_combine_simple(matrix_type):
    m1 = matrix_type(np.array(range(9)).reshape((3,3)))

    # Combine :
    # [0, 1, 2              [0, 1, 2                [ 0,  2,  4
    #  3, 4, 5      and      3, 4, 5        gives     6,  8, 10,
    #  6, 7, 8]              6, 7, 8]                12, 14, 16

    np.testing.assert_array_equal([[0,2,4], [6,8,10], [12,14,16]], m1.combine(m1, (3, 3)).as_dense())


@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_combine_columns(matrix_type):
    m1 = matrix_type(np.array(range(9)).reshape((3, 3)))
    m2 = matrix_type(np.array(range(6)).reshape((3, 2)))

    # Combine :
    # [0, 1, 2              [0, 1,                [ 0,  1,  2,  1,
    #  3, 4, 5      and      2, 3,        gives     5,  4,  5,  3
    #  6, 7, 8]              4, 5]                 10,  7,  8,  5
    #                        |  |
    #                        v  v
    #                        0  3

    np.testing.assert_array_equal([[0,1,2,1], [5,4,5,3], [10,7,8,5]], m1.combine(m2, (3, 4), col_indices_map={0:0, 1:3}).as_dense())


@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_combine_rows(matrix_type):
    m1 = matrix_type(np.array(range(9)).reshape((3, 3)))
    m2 = matrix_type(np.array(range(6)).reshape((2,3)))

    # Combine :
    # [0, 1, 2              [0, 1, 2  -> 0                  [ 0,  2,  4,
    #  3, 4, 5      and      3, 4, 5] -> 3        gives       3,  4,  5,
    #  6, 7, 8]                                               6,  7,  8,
    #                                                         3,  4,  5

    np.testing.assert_array_equal([[0,2,4], [3,4,5], [6,7,8], [3,4,5]], m1.combine(m2, (4, 3), row_indices_map={0:0, 1:3}).as_dense())


@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_combine_rows_and_columns(matrix_type):
    m1 = matrix_type(np.array(range(9)).reshape((3, 3)))
    m2 = matrix_type(np.array(range(4)).reshape((2,2)))

    # Combine :
    # [0, 1, 2              [0, 1,  -> 0                  [ 0,  1,  2,  1,
    #  3, 4, 5      and      2, 3]  -> 3        gives       3,  4,  5,  0,
    #  6, 7, 8]              |  |                           6,  7,  8,  0,
    #                        v  v                           2,  0,  0,  3]
    #                        0  3

    np.testing.assert_array_equal([[0,1,2,1], [3,4,5,0], [6,7,8,0], [2,0,0,3]],
                                  m1.combine(m2, (4, 4), row_indices_map={0:0, 1:3}, col_indices_map={0:0, 1:3}).as_dense())

    # Combine :
    # [0, 1, 2              [0, 1,  -> 3                  [ 0,  1,  2,  -,
    #  3, 4, 5      and      2, 3]  -> 4        gives       3,  4,  5,  -,
    #  6, 7, 8]              |  |                           6,  7,  8,  -,
    #                        v  v                           0,  -,  -,  1,
    #                        0  3                           2,  -,  -,  3]

    np.testing.assert_array_equal([[0,1,2,0], [3,4,5,0], [6,7,8,0], [0,0,0,1],[2,0,0,3]],
                                  m1.combine(m2, (5, 4), row_indices_map={0:3, 1:4}, col_indices_map={0:0, 1:3}).as_dense())



########################
# Exceptions
def test_exception_wrong_matrix_type():
    with pytest.raises(mangoes.utils.exceptions.UnsupportedType):
        mangoes.utils.arrays.Matrix.factory("xxx")


def test_exception_compare_non_sparse_matrices():
    with pytest.raises(mangoes.utils.exceptions.IncompatibleValue):
        mangoes.utils.arrays.csrSparseMatrix.allclose(3, 4)
