import numpy as np

import pytest
import scipy

import mangoes
import mangoes.evaluation.similarity


def get_representation(matrix_type):
    #   |
    #   |
    #   |
    #   f    e    d
    #   |
    #   |
    #   |
    #   |         b c
    #   |---------a---------
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    matrix = matrix_type([[1, 0],
                          [1, 0.1],
                          [1.1, 0.1],
                          [1, 1],
                          [0.5, 1],
                          [0, 1]])
    return mangoes.base.Embeddings(words, matrix)


# We consider 2 datasets :
# The first one fits with the representation :
dataset1 = mangoes.evaluation.similarity.Dataset("dataset1", ['a b 1.0', 'a c 1.0', 'a d 0.71', 'a e 0.45', 'a f 0.0'])
# The second one doesn't :
dataset2 = mangoes.evaluation.similarity.Dataset("dataset2", ['b f 0.9', 'b c 0.8', 'b e 0.5', 'b d 0.2'])


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_similarity_prediction(matrix_type):
    embedding = get_representation(matrix_type)

    evaluation = mangoes.evaluation.similarity.Evaluator(embedding)

    assert 1.0 == evaluation.predict(('a', 'a'))

    np.testing.assert_almost_equal(0.99, evaluation.predict(('a', 'b')), decimal=2)
    np.testing.assert_almost_equal(0.99, evaluation.predict(('a', 'c')), decimal=2)
    np.testing.assert_almost_equal(np.sqrt(2) / 2, evaluation.predict(('a', 'd')), decimal=2)
    assert 0.0 == evaluation.predict(('a', 'f'))


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_similarity_dataset1(matrix_type):
    embedding = get_representation(matrix_type)

    evaluation = mangoes.evaluation.similarity.Evaluation(embedding, dataset1)

    # score
    score = evaluation.get_score()
    assert 5 == score.nb
    assert 0.99 < score.pearson.coeff
    assert 0.97 < score.spearman.coeff

    expected_summary_report = """
                                                                          pearson       spearman
                                                      Nb questions        (p-val)        (p-val)
================================================================================================
dataset1                                                       5/5     1.0(7e-10)   0.975(5e-03)
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == evaluation.get_report()
    assert expected_summary_report == str(evaluation.get_report())

    # detail : with questions and predictions
    expected_detail_report = """
                                                                          pearson       spearman
                                                      Nb questions        (p-val)        (p-val)
================================================================================================
dataset1                                                       5/5     1.0(7e-10)   0.975(5e-03)

                                               gold          score                              
a b                                             1.0            1.0
a c                                             1.0            1.0
a d                                            0.71           0.71
a e                                            0.45           0.45
a f                                             0.0            0.0

------------------------------------------------------------------------------------------------
"""

    assert expected_detail_report == evaluation.get_report(show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_similarity_dataset2(matrix_type):
    embedding = get_representation(matrix_type)

    evaluation = mangoes.evaluation.similarity.Evaluation(embedding, dataset2)

    # score
    score = evaluation.get_score()
    assert 4 == score.nb
    assert 0.5 > np.abs(score.pearson.coeff)
    assert 0.5 > np.abs(score.spearman.coeff)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_similarity_with_2_datasets(matrix_type):
    embedding = get_representation(matrix_type)

    evaluation = mangoes.evaluation.similarity.Evaluation(embedding, dataset1, dataset2)

    # SCORES
    score_dataset1 = evaluation.get_score("dataset1")
    assert 5 == score_dataset1.nb
    assert 0.99 < score_dataset1.pearson.coeff
    assert 0.97 < score_dataset1.spearman.coeff

    score_dataset2 = evaluation.get_score("dataset2")
    assert 4 == score_dataset2.nb
    assert 0.5 > np.abs(score_dataset2.pearson.coeff)
    assert 0.5 > np.abs(score_dataset2.spearman.coeff)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_similarity_with_subsets(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.similarity.Dataset("dataset", {"subset1": dataset1.data,
                                                                "subset2": dataset2.data})

    evaluation = mangoes.evaluation.similarity.Evaluation(embedding, dataset)

    # SCORES
    score_subset1 = evaluation.get_score("dataset/subset1")
    assert 5 == score_subset1.nb
    assert 0.99 < score_subset1.pearson.coeff
    assert 0.97 < score_subset1.spearman.coeff

    score_subset2 = evaluation.get_score("dataset/subset2")
    assert 4 == score_subset2.nb
    assert 0.5 > np.abs(score_subset2.pearson.coeff)
    assert 0.5 > np.abs(score_subset2.spearman.coeff)

    score_dataset = evaluation.get_score("dataset")
    assert 9 == score_dataset.nb


#############
# OOV
@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_similarity_with_oov(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.similarity.Dataset("dataset", dataset1.data + ['x y 0.3'])

    evaluation = mangoes.evaluation.similarity.Evaluation(embedding, dataset)

    assert 1 == evaluation._questions_by_subset["dataset"].nb_oov

    # SCORE
    score = evaluation.get_score("dataset")
    expected_score = mangoes.evaluation.similarity.Evaluation(embedding, dataset1).get_score()
    assert 5 == score.nb
    assert expected_score == score


#############
# Duplicates
@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_similarity_duplicates(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.similarity.Dataset("dataset", dataset1.data + dataset1.data[:2])

    evaluation = mangoes.evaluation.similarity.Evaluation(embedding, dataset)

    assert 2 == evaluation._questions_by_subset["dataset"].nb_duplicates

    # score
    score_with_duplicate = evaluation.get_score(keep_duplicates=True)
    assert 7 == score_with_duplicate.nb

    score_no_duplicate = evaluation.get_score(keep_duplicates=False)
    assert 5 == score_no_duplicate.nb

    assert score_with_duplicate.pearson.coeff != score_no_duplicate.pearson.coeff
    assert score_with_duplicate.spearman.coeff != score_no_duplicate.spearman.coeff


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_similarity_duplicates_with_different_gold(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.similarity.Dataset("dataset", dataset1.data + ['a b 0.8', 'a c 0.9'])

    evaluation = mangoes.evaluation.similarity.Evaluation(embedding, dataset)

    assert 0 == evaluation._questions_by_subset["dataset"].nb_duplicates

    # score
    score_with_duplicate = evaluation.get_score(keep_duplicates=True)
    assert 7 == score_with_duplicate.nb

    score_no_duplicate = evaluation.get_score(keep_duplicates=False)
    assert 7 == score_no_duplicate.nb

    assert score_with_duplicate.spearman == score_no_duplicate.spearman
    np.testing.assert_almost_equal(score_with_duplicate.pearson.coeff, score_no_duplicate.pearson.coeff)
    # there may be non deterministic rounding issues with pearson


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_similarity_duplicates_with_subsets(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.similarity.Dataset("dataset", {"subset1": ['a b 0.5', 'a c 0.5', 'b c 0.25'],
                                                                "subset2": ['a b 0.4', 'a c 0.5', 'b c 0.25']})

    evaluation = mangoes.evaluation.similarity.Evaluation(embedding, dataset)

    assert 0 == evaluation._questions_by_subset["dataset/subset1"].nb_duplicates
    assert 0 == evaluation._questions_by_subset["dataset/subset2"].nb_duplicates
    assert 2 == evaluation._questions_by_subset["dataset"].nb_duplicates

    # score
    score_with_duplicate = evaluation.get_score(keep_duplicates=True)
    assert 6 == score_with_duplicate.nb

    score_no_duplicate = evaluation.get_score(keep_duplicates=False)
    assert 4 == score_no_duplicate.nb

    assert score_with_duplicate.pearson.coeff != score_no_duplicate.pearson.coeff
    assert score_with_duplicate.spearman.coeff != score_no_duplicate.spearman.coeff


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_similarity_lower(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.similarity.Dataset("dataset",
                                                    ['a b 1.0', 'a c 1.0', 'A D 0.71', 'A e 0.45', 'a f 0.0'])

    # LOWER = True
    evaluation = mangoes.evaluation.similarity.Evaluation(embedding, dataset, lower=True)

    # score
    score = evaluation.get_score()
    assert 5 == score.nb
    assert 0.99 < score.pearson.coeff
    assert 0.97 < score.spearman.coeff

    expected_summary_report = """
                                                                          pearson       spearman
                                                      Nb questions        (p-val)        (p-val)
================================================================================================
dataset                                                        5/5     1.0(7e-10)   0.975(5e-03)
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == evaluation.get_report()
    assert expected_summary_report == str(evaluation.get_report())

    # detail : with questions and predictions
    expected_detail_report = """
                                                                          pearson       spearman
                                                      Nb questions        (p-val)        (p-val)
================================================================================================
dataset                                                        5/5     1.0(7e-10)   0.975(5e-03)

                                               gold          score                              
a b                                             1.0            1.0
a c                                             1.0            1.0
a d                                            0.71           0.71
a e                                            0.45           0.45
a f                                             0.0            0.0

------------------------------------------------------------------------------------------------
"""

    assert expected_detail_report == evaluation.get_report(show_questions=True)

    # LOWER = False
    evaluation = mangoes.evaluation.similarity.Evaluation(embedding, dataset, lower=False)

    assert 3 == evaluation.get_score().nb
    expected_summary_report = """
                                                                          pearson       spearman
                                                      Nb questions        (p-val)        (p-val)
================================================================================================
dataset                                                        3/5     1.0(5e-04)   0.866(3e-01)
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == evaluation.get_report()
    assert expected_summary_report == str(evaluation.get_report())

    # detail : with questions and predictions
    expected_detail_report = """
                                                                          pearson       spearman
                                                      Nb questions        (p-val)        (p-val)
================================================================================================
dataset                                                        3/5     1.0(5e-04)   0.866(3e-01)

                                               gold          score                              
a b                                             1.0            1.0
a c                                             1.0            1.0
a f                                             0.0            0.0

------------------------------------------------------------------------------------------------
"""

    assert expected_detail_report == evaluation.get_report(show_questions=True)
