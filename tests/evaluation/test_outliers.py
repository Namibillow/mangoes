import numpy as np

import pytest
import scipy

import mangoes
import mangoes.evaluation.outlier


def get_representation(matrix_type):
    #   |   e
    #   |  f d
    #   |
    #   |
    #   |
    #   |
    #   |               a
    #   |              b c
    #   |------------------------
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e', 'f'])
    matrix = matrix_type([[1.0, 0.2],
                          [0.9, 0.1],
                          [1.1, 0.1],
                          [0.3, 0.9],
                          [0.2, 1.0],
                          [0.1, 0.9]])
    return mangoes.base.Embeddings(words, matrix)


# We consider 2 datasets :
# The first one fits with the representation :
dataset1 = mangoes.evaluation.outlier.Dataset("dataset1", ['a b c d', 'd e f a'])
# The second one doesn't :
dataset2 = mangoes.evaluation.outlier.Dataset("dataset2", ['a b e c', 'd e c f'])


def test_user_dataset():
    dataset = mangoes.evaluation.outlier.Dataset("My dataset", ['a b c d', 'e f g h'])
    assert ['a b c d', 'e f g h'] == dataset.data


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_outlier_predict_position(matrix_type):
    embedding = get_representation(matrix_type)
    evaluation = mangoes.evaluation.outlier.Evaluator(embedding)

    # dataset1
    assert 4 == evaluation.predict('a b c d')
    assert 4 == evaluation.predict('d e f a')

    # dataset2
    assert 3 == evaluation.predict('a b e c')
    assert 3 == evaluation.predict('d e c f')

    # assert 1 == evaluation.predict('c d e')
    assert 5 == evaluation.predict('a b c d e')
    assert 3 == evaluation.predict('a b e c')


##################
# Evaluation
@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_outlier_detection_dataset1(matrix_type):
    embedding = get_representation(matrix_type)

    evaluation = mangoes.evaluation.outlier.Evaluation(embedding, dataset1, lower=False)

    # PREDICTIONS
    assert {'a b c d': 4,
            'd e f a': 4} == evaluation.predictions

    # SCORE
    score = evaluation.get_score()

    assert 1 == score.opp
    assert 1 == score.accuracy
    assert 2 == score.nb

    # REPORTS
    expected_summary_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset1                                                             2/2     100.00%     100.00%
------------------------------------------------------------------------------------------------
"""
    assert expected_summary_report == evaluation.get_report()

    expected_detail_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset1                                                             2/2     100.00%     100.00%

                                                                                outlier position
a b c d                                                                                        4
d e f a                                                                                        4

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_outlier_detection_dataset2(matrix_type):
    embedding = get_representation(matrix_type)

    # 'a b e c' -> returns 'a b c e' -> outlier position = 3 / acc = 0 / OPP = 3/4
    # 'd e c f' -> returns 'd e f c' -> outlier position = 3 / acc = 0 / OPP = 3/4
    # Total : acc = 0 / OPP = (3/4 + 3/4)/2 = 3 / 4

    evaluation = mangoes.evaluation.outlier.Evaluation(embedding, dataset2)
    score = evaluation.get_score()

    assert 0 == score.accuracy
    assert 3 / 4 == score.opp
    assert 2 == score.nb


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_outlier_detection_with_2_datasets(matrix_type):
    embedding = get_representation(matrix_type)

    evaluation = mangoes.evaluation.outlier.Evaluation(embedding, dataset1, dataset2)

    # SCORES
    score_dataset1 = evaluation.get_score("dataset1")
    assert 1 == score_dataset1.accuracy
    assert 1 == score_dataset1.opp
    assert 2 == score_dataset1.nb

    score_dataset2 = evaluation.get_score("dataset2")
    assert 0 == score_dataset2.accuracy
    assert 3 / 4 == score_dataset2.opp
    assert 2 == score_dataset2.nb

    # REPORTS
    expected_summary_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset1                                                             2/2     100.00%     100.00%
------------------------------------------------------------------------------------------------
dataset2                                                             2/2      75.00%       0.00%
------------------------------------------------------------------------------------------------
"""
    assert expected_summary_report == evaluation.get_report()

    expected_detail_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset1                                                             2/2     100.00%     100.00%

                                                                                outlier position
a b c d                                                                                        4
d e f a                                                                                        4

------------------------------------------------------------------------------------------------
dataset2                                                             2/2      75.00%       0.00%

                                                                                outlier position
a b e c                                                                                        3
d e c f                                                                                        3

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_outlier_detection_with_subsets(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.outlier.Dataset("dataset", {"subset1": dataset1.data,
                                                             "subset2": dataset2.data})

    evaluation = mangoes.evaluation.outlier.Evaluation(embedding, dataset)
    score = evaluation.get_score()

    assert 4 == score.nb
    assert 0.5 == score.accuracy
    assert (1 + 1 + 3 / 4 + 3 / 4) / 4 == score.opp

    expected_summary_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset                                                              4/4      87.50%      50.00%
------------------------------------------------------------------------------------------------
"""
    assert expected_summary_report == evaluation.get_report()

    expected_report_with_subsets = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset                                                              4/4      87.50%      50.00%

    subset1                                                          2/2     100.00%     100.00%
    subset2                                                          2/2      75.00%       0.00%

------------------------------------------------------------------------------------------------
"""
    assert expected_report_with_subsets == evaluation.get_report(show_subsets=True)

    expected_detail_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset                                                              4/4      87.50%      50.00%

    subset1                                                          2/2     100.00%     100.00%

                                                                                outlier position
    a b c d                                                                                    4
    d e f a                                                                                    4

    subset2                                                          2/2      75.00%       0.00%

                                                                                outlier position
    a b e c                                                                                    3
    d e c f                                                                                    3

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_outlier_detection_with_oov(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.outlier.Dataset("dataset", ['a b c d', 'a b c x', 'x y z t'])

    evaluation = mangoes.evaluation.outlier.Evaluation(embedding, dataset)

    assert 2 == evaluation._questions_by_subset["dataset"].nb_oov

    score = evaluation.get_score()

    assert 1 == score.nb
    assert 1 == score.accuracy
    assert 1 == score.opp


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_outlier_detection_all_oov(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.outlier.Dataset("dataset", ['a b c x', 'a b y d', 'x y z t'])

    evaluation = mangoes.evaluation.outlier.Evaluation(embedding, dataset)

    assert 3 == evaluation._questions_by_subset["dataset"].nb_oov

    score = evaluation.get_score()

    assert 0 == score.nb
    assert np.isnan(score.opp)
    assert np.isnan(score.accuracy)

    expected_summary_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset                                                              0/3          NA          NA
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == evaluation.get_report()

    # detail : with questions and predictions
    expected_detail_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset                                                              0/3          NA          NA

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_outlier_detection_with_duplicates(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.outlier.Dataset("dataset", ['a b c d', 'a b c d', 'd e a f'])

    evaluation = mangoes.evaluation.outlier.Evaluation(embedding, dataset)

    assert 1 == evaluation._questions_by_subset["dataset"].nb_duplicates

    # PREDICTIONS
    assert {'a b c d': 4,
            'd e a f': 3} == evaluation.predictions

    #####################
    # keep duplicates
    # SCORE
    score = evaluation.get_score(keep_duplicates=True)
    assert 3 == score.nb
    assert 2 / 3 == score.accuracy
    assert (1 + 1 + 3 / 4) / 3 == score.opp

    # REPORTS
    expected_summary_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset                                                              3/3      91.67%      66.67%
                                                 (including 1 duplicate)
------------------------------------------------------------------------------------------------
"""
    assert expected_summary_report == evaluation.get_report(keep_duplicates=True)

    expected_detail_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset                                                              3/3      91.67%      66.67%
                                                 (including 1 duplicate)

                                                                                outlier position
a b c d                                                                                        4
a b c d                                                                                        4
d e a f                                                                                        3

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(show_questions=True)

    #####################
    # remove duplicates
    # SCORE
    score = evaluation.get_score(keep_duplicates=False)
    assert 2 == score.nb
    assert 1 / 2 == score.accuracy
    assert (1 + 3 / 4) / 2 == score.opp

    # REPORTS
    expected_summary_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset                                                              2/3      87.50%      50.00%
                                                          (-1 duplicate)
------------------------------------------------------------------------------------------------
"""
    assert expected_summary_report == evaluation.get_report(keep_duplicates=False)

    expected_detail_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset                                                              2/3      87.50%      50.00%
                                                          (-1 duplicate)

                                                                                outlier position
a b c d                                                                                        4
d e a f                                                                                        3

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(keep_duplicates=False, show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_outlier_detection_lower(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.outlier.Dataset("dataset", ['A B C D', 'd e a f', 'A b C e'])

    # LOWER = False
    evaluation = mangoes.evaluation.outlier.Evaluation(embedding, dataset, lower=False)

    assert {'d e a f': 3} == evaluation.predictions

    score = evaluation.get_score()
    assert 1 == score.nb
    assert 0 == score.accuracy
    assert 3 / 4 == score.opp

    expected_detail_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset                                                              1/3      75.00%       0.00%

                                                                                outlier position
d e a f                                                                                        3

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(keep_duplicates=False, show_questions=True)

    # LOWER = True
    evaluation = mangoes.evaluation.outlier.Evaluation(embedding, dataset, lower=True)
    assert {'a b c d': 4,
            'd e a f': 3,
            'a b c e': 4} == evaluation.predictions

    score = evaluation.get_score()
    assert 3 == score.nb
    assert 2 / 3 == score.accuracy
    assert (1 + 3 / 4 + 1) / 3 == score.opp

    expected_detail_report = """
                                                            Nb questions         OPP    accuracy
================================================================================================
dataset                                                              3/3      91.67%      66.67%

                                                                                outlier position
a b c d                                                                                        4
d e a f                                                                                        3
a b c e                                                                                        4

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(keep_duplicates=False, show_questions=True)
