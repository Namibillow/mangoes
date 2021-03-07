import numpy as np
import pytest
from scipy import sparse

import mangoes.utils
import mangoes.weighting
import mangoes.utils.arrays


@pytest.mark.unittest
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_joint_probabilities(matrix_type):
    matrix = matrix_type([[1, 2, 4, 0], [3, 1, 1, 0]])

    actual_result = mangoes.weighting.JointProbabilities()(matrix)
    expected_result = matrix_type([[1 / 12, 1 / 6, 1 / 3, 0], [1 / 4, 1 / 12, 1 / 12, 0]], dtype=np.float64)

    try:
        assert np.allclose(expected_result, actual_result)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result, actual_result)


@pytest.mark.unittest
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_conditional_probabilities(matrix_type):
    matrix = mangoes.utils.arrays.Matrix.factory(matrix_type([[1, 2, 4, 0], [3, 1, 1, 0]]))

    actual_result = mangoes.weighting.ConditionalProbabilities()(matrix)
    expected_result = matrix_type([[1 / 7, 2 / 7, 4 / 7, 0], [3 / 5, 1 / 5, 1 / 5, 0]])

    try:
        assert np.allclose(expected_result, actual_result)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result, actual_result)


@pytest.mark.unittest
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_probabilities_ratio(matrix_type):
    matrix = matrix_type([[1, 2, 4, 0], [3, 1, 1, 0]])

    alpha = 1
    actual_result = mangoes.weighting.ProbabilitiesRatio(alpha=alpha)(matrix)

    expected_result = matrix_type([[3 / 7, 8 / 7, 48 / 35, 0], [9 / 5, 4 / 5, 12 / 25, 0]])

    try:
        assert np.allclose(expected_result, actual_result)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result, actual_result)

    alpha = 3
    actual_result = mangoes.weighting.ProbabilitiesRatio(alpha=alpha)(matrix)

    expected_result = matrix_type([
        [(1 / 7) / ((4 ** 3) / 216),
         (2 / 7) / ((3 ** 3) / 216), (4 / 7) / ((5 ** 3) / 216), 0
         ],
        [(3 / 5) / ((4 ** 3) / 216), (1 / 5) / ((3 ** 3) / 216),
         (1 / 5) / ((5 ** 3) / 216), 0
         ]
    ])

    try:
        assert np.allclose(expected_result, actual_result)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result, actual_result)


@pytest.mark.unittest
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_pmi(matrix_type):
    matrix = matrix_type([[1, 2, 4, 0], [3, 1, 1, 0]])

    alpha = 1
    actual_result = mangoes.weighting.PMI(alpha=alpha)(matrix)

    expected_result = matrix_type([[np.log(3 / 7), np.log(8 / 7), np.log(48 / 35), 0],
                                   [np.log(9 / 5), np.log(4 / 5), np.log(12 / 25), 0]])

    try:
        assert np.allclose(expected_result, actual_result)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result, actual_result)


    alpha = 2
    actual_result = mangoes.weighting.PMI(alpha=alpha)(matrix)

    # PMI = log( p(w,c) / [ p(w) * p_alpha(c) ] )
    # #w = sum(#w_i) = #c = sum(#c_i) = #(w,c) = 1+2+4+3+1+1 = 12

    # p(w0,c0) = #(w0,c0) / #(w,c) = 1 / 12 = 1/12

    # p(w0) = #w0 / #w = (1+2+4) / 12 = 7/12

    # p(c0) = #c0 / sum(#c_i) = 4 / (4 + 3 + 5)
    # p_alpha(c0) = (#c0)**alpha / sum((#c_i)**alpha) = 4**2 / (4**2 + 3**2 + 5**2) = 16/50

    # PMI(w0,c0) = log(p(w0,c0)  / [p(w0) * p_alpha(c0)]
    expected_first_cell = np.log((1 / 12) / ((7 / 12) * (16/50)))
    np.testing.assert_almost_equal(expected_first_cell, actual_result[0, 0])

    expected_result = matrix_type([[-0.80647586586694853, 0.46203545959655862, 0.13353139262452257, 0],
                                   [ 0.62860865942237409,  0.10536051565782635, -0.91629073187415511,  0]])
    # [[np.log((1/12)/((7/12)*(16/50))), np.log((2/12)/((7/12)*(9/50))), np.log((4/12)/((7/12)*(25/50))), 0],
    #  [np.log((3/12)/((5/12)*(16/50))), np.log((1/12)/((5/12)*(9/50))), np.log((1/12)/((5/12)*(25/50))), 0]])

    try:
        assert np.allclose(expected_result, actual_result)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result, actual_result)


@pytest.mark.unittest
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_ppmi(matrix_type):
    matrix = matrix_type([[1, 2, 4, 0], [3, 1, 1, 0]])

    alpha = 1
    actual_result = mangoes.weighting.PPMI(alpha=alpha)(matrix)

    expected_result = matrix_type([[0.0, np.log(8 / 7), np.log(48 / 35), 0.0],
                                   [np.log(9 / 5), 0.0, 0.0, 0.0]])

    try:
        assert np.allclose(expected_result, actual_result)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result, actual_result)

    alpha = 2
    actual_result = mangoes.weighting.PPMI(alpha=alpha)(matrix)

    expected_result = matrix_type([[0,                   0.46203545959655862, 0.13353139262452257, 0],
                                   [0.62860865942237409, 0.10536051565782635, 0,                   0]])

    try:
        assert np.allclose(expected_result, actual_result)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result, actual_result)


@pytest.mark.unittest
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_sppmi(matrix_type):
    matrix = matrix_type([[1, 2, 4, 0], [3, 1, 1, 0]])

    alpha = 1
    shift = 5.0 / 4.0

    actual_result = mangoes.weighting.ShiftedPPMI(shift=shift, alpha=alpha)(matrix)

    expected_result = matrix_type([[0.0, 0.0, np.log(48 / 35 / shift), 0.0],
                                   [np.log(9 / 5 / shift), 0.0, 0.0, 0.0]])

    try:
        assert np.allclose(expected_result, actual_result)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result, actual_result)

    alpha = 2
    shift = 5.0 / 4.0 # log(shift) = 0.22314355131420976

    actual_result = mangoes.weighting.ShiftedPPMI(shift=shift, alpha=alpha)(matrix)

    expected_result = matrix_type([[0,                   0.23889190828234885, 0, 0],
                                   [0.40546510810816433, 0,                   0, 0]])

    try:
        assert np.allclose(expected_result, actual_result)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result, actual_result)


@pytest.mark.unittest
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_tfidf(matrix_type):
    matrix = matrix_type([[1, 2, 4, 0],
                          [3, 1, 1, 0],
                          [0, 0, 0, 0]])

    actual_result = mangoes.weighting.TFIDF()(matrix)

    expected_result = matrix_type([[1 / 7 * np.log(4 / 3), 2 / 7 * np.log(4 / 3), 4 / 7 * np.log(4 / 3), 0],
                                   [3 / 5 * np.log(4 / 3), 1 / 5 * np.log(4 / 3), 1 / 5 * np.log(4 / 3), 0],
                                   [0,                     0,                     0,                     0]])

    try:
        assert np.allclose(expected_result, actual_result)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected_result, actual_result)


########################
# Exceptions
def test_exception_negative_alpha():
    with pytest.raises(mangoes.utils.exceptions.NotAllowedValue):
        mangoes.weighting.PMI(alpha=-1)(None)


def test_exception_shift_lower_than_one():
    with pytest.raises(mangoes.utils.exceptions.NotAllowedValue):
        mangoes.weighting.ShiftedPPMI(shift=0.5)(None)
