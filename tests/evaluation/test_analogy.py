import pytest
import scipy.sparse
import numpy as np

import mangoes.evaluation.base
import mangoes.evaluation.analogy


def get_representation(matrix_type):
    #   |
    #   |
    #   |
    #   d
    #   c     f
    #   |     e
    #   |
    #   |         b
    #   |---------a---------
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e', 'f'])
    matrix = matrix_type([[1, 0],
                          [1, 0.2],
                          [0, 1],
                          [0, 1.2],
                          [0.7, 0.7],
                          [0.7, 0.8]])
    return mangoes.base.Embeddings(words, matrix)


# We consider 2 datasets :
# The first one fits best with the representation using cosadd:
dataset1 = mangoes.evaluation.analogy.Dataset("dataset1", ['a b c d', 'a b e f', ' a e b f', 'b f e c'])
# The second one fits best with cosmul: # TODO: explain this
dataset2 = mangoes.evaluation.analogy.Dataset("dataset2", ['d f c e', 'd f e a'])


def test_datasets():
    assert [('a b c', 'd'), ('a b e', 'f'), ('a e b', 'f'), ('b f e', 'c')] == dataset1.data


##################
# Evaluator

@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_prediction(matrix_type):
    embedding = get_representation(matrix_type)

    evaluator = mangoes.evaluation.analogy.Evaluator(embedding)

    assert 'd' == evaluator.predict('a b c').using_cosadd[0]
    assert 'f' == evaluator.predict('a b e').using_cosadd[0]
    assert 'f' == evaluator.predict('a e b').using_cosadd[0]
    assert 'c' == evaluator.predict('b f e').using_cosadd[0]

    assert 'e' == evaluator.predict('d f c').using_cosadd[0]
    assert 'b' == evaluator.predict('d f e').using_cosadd[0]  #

    assert 'd' == evaluator.predict('a b c').using_cosmul[0]
    assert 'c' == evaluator.predict('a b e').using_cosmul[0]  #
    assert 'c' == evaluator.predict('a e b').using_cosmul[0]  #
    assert 'c' == evaluator.predict('b f e').using_cosmul[0]

    assert 'e' == evaluator.predict('d f c').using_cosmul[0]
    assert 'a' == evaluator.predict('d f e').using_cosmul[0]


##################
# Evaluation
@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy(matrix_type):
    embedding = get_representation(matrix_type)

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset1)

    # SCORES
    score = evaluation.get_score()
    assert 4 == score.nb
    assert 1.0 == score.cosadd
    assert 0.5 == score.cosmul

    # REPORTS
    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset1                                                             4/4     100.00%      50.00%
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == evaluation.get_report()
    assert expected_summary_report == str(evaluation.get_report())

    expected_detail_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset1                                                             4/4     100.00%      50.00%

a b c d                                                                            d           d
a b e f                                                                            f           c
a e b f                                                                            f           c
b f e c                                                                            c           c

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_with_2_datasets(matrix_type):
    embedding = get_representation(matrix_type)

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset1, dataset2)

    # SCORES
    score_dataset1 = evaluation.get_score("dataset1")
    assert 4 == score_dataset1.nb
    assert 1.0 == score_dataset1.cosadd
    assert 0.5 == score_dataset1.cosmul

    score_dataset2 = evaluation.get_score("dataset2")
    assert 2 == score_dataset2.nb
    assert 0.5 == score_dataset2.cosadd
    assert 1.0 == score_dataset2.cosmul

    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset1                                                             4/4     100.00%      50.00%
------------------------------------------------------------------------------------------------
dataset2                                                             2/2      50.00%     100.00%
------------------------------------------------------------------------------------------------
"""
    assert expected_summary_report == str(evaluation.get_report())
    assert expected_summary_report == evaluation.get_report()

    # detail : with questions and predictions
    expected_detail_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset1                                                             4/4     100.00%      50.00%

a b c d                                                                            d           d
a b e f                                                                            f           c
a e b f                                                                            f           c
b f e c                                                                            c           c

------------------------------------------------------------------------------------------------
dataset2                                                             2/2      50.00%     100.00%

d f c e                                                                            e           e
d f e a                                                                            b           a

------------------------------------------------------------------------------------------------
"""

    assert expected_detail_report == evaluation.get_report(show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_with_subset(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.analogy.Dataset("dataset", {"subset1": dataset1.data,
                                                             "subset2": dataset2.data})

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset)

    # SCORES
    score = evaluation.get_score()
    assert 6 == score.nb
    assert 5 / 6 == score.cosadd
    assert 4 / 6 == score.cosmul

    score_subset1 = evaluation.get_score("dataset/subset1")
    assert 4 == score_subset1.nb
    assert 1 == score_subset1.cosadd
    assert 0.5 == score_subset1.cosmul

    score_subset2 = evaluation.get_score("dataset/subset2")
    assert 2 == score_subset2.nb
    assert 0.5 == score_subset2.cosadd
    assert 1 == score_subset2.cosmul

    # REPORTS
    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              6/6      83.33%      66.67%
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == str(evaluation.get_report())

    # with subsets
    expected_subset_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              6/6      83.33%      66.67%

    subset1                                                          4/4     100.00%      50.00%
    subset2                                                          2/2      50.00%     100.00%

------------------------------------------------------------------------------------------------
"""
    assert expected_subset_report == evaluation.get_report(show_subsets=True)

    # with subsets and predictions
    expected_subset_and_questions_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              6/6      83.33%      66.67%

    subset1                                                          4/4     100.00%      50.00%

    a b c d                                                                        d           d
    a b e f                                                                        f           c
    a e b f                                                                        f           c
    b f e c                                                                        c           c

    subset2                                                          2/2      50.00%     100.00%

    d f c e                                                                        e           e
    d f e a                                                                        b           a

------------------------------------------------------------------------------------------------
"""
    assert expected_subset_and_questions_report == evaluation.get_report(show_subsets=True, show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_with_subsubset(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.analogy.Dataset("dataset", {"subset1": {"ss11": dataset1.data[:2],
                                                                         "ss12": dataset1.data[2:]},
                                                             "subset2": dataset2.data})

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset)

    # SCORES
    score = evaluation.get_score()

    assert 6 == score.nb
    assert 5 / 6 == score.cosadd
    assert 4 / 6 == score.cosmul

    # REPORTS
    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              6/6      83.33%      66.67%
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == str(evaluation.get_report())

    # with subsets
    expected_subset_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              6/6      83.33%      66.67%

    subset1                                                          4/4     100.00%      50.00%

        ss11                                                         2/2     100.00%      50.00%
        ss12                                                         2/2     100.00%      50.00%

    subset2                                                          2/2      50.00%     100.00%

------------------------------------------------------------------------------------------------
"""
    assert expected_subset_report == evaluation.get_report(show_subsets=True)

    # with subsets and predictions
    expected_subset_and_questions_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              6/6      83.33%      66.67%

    subset1                                                          4/4     100.00%      50.00%

        ss11                                                         2/2     100.00%      50.00%

        a b c d                                                                    d           d
        a b e f                                                                    f           c

        ss12                                                         2/2     100.00%      50.00%

        a e b f                                                                    f           c
        b f e c                                                                    c           c

    subset2                                                          2/2      50.00%     100.00%

    d f c e                                                                        e           e
    d f e a                                                                        b           a

------------------------------------------------------------------------------------------------
"""
    assert expected_subset_and_questions_report == evaluation.get_report(show_questions=True)


#############
# OOV
@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_with_oov(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.analogy.Dataset("dataset", ['a b c d', 'a b e x', 'x y z t'])

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset)

    assert 2 == evaluation._questions_by_subset["dataset"].nb_oov

    # SCORE
    score = evaluation.get_score()

    assert 1 == score.nb
    assert 1.0 == score.cosadd
    assert 1.0 == score.cosmul

    # REPORTS
    # summary = string representation
    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              1/3     100.00%     100.00%
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == evaluation.get_report()

    # detail : with questions and predictions
    expected_detail_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              1/3     100.00%     100.00%

a b c d                                                                            d           d

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_all_oov(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.analogy.Dataset("dataset", ['a b x d', 'a b e y', 'x y z t'])

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset)

    # SCORE
    score = evaluation.get_score()

    assert 0 == score.nb
    assert np.isnan(score.cosadd)
    assert np.isnan(score.cosmul)

    # REPORT
    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              0/3          NA          NA
------------------------------------------------------------------------------------------------
"""
    assert expected_summary_report == evaluation.get_report()


#############
# Duplicates
@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_duplicates(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.analogy.Dataset("dataset", ['a b c d', 'a b e f', 'a\tb\tc\td'])

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset)

    assert 1 == evaluation._questions_by_subset["dataset"].nb_duplicates

    # ################
    # KEEP DUPLICATES
    # SCORE
    score_with_duplicate = evaluation.get_score(keep_duplicates=True)

    assert 3 == score_with_duplicate.nb
    assert 1.0 == score_with_duplicate.cosadd
    assert 2 / 3 == score_with_duplicate.cosmul

    # REPORTS
    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              3/3     100.00%      66.67%
                                                 (including 1 duplicate)
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == evaluation.get_report(keep_duplicates=True)

    expected_detail_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              3/3     100.00%      66.67%
                                                 (including 1 duplicate)

a b c d                                                                            d           d
a b e f                                                                            f           c
a b c d                                                                            d           d

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(show_questions=True)

    # ################
    # REMOVE DUPLICATES
    # SCORE
    score_no_duplicate = evaluation.get_score(keep_duplicates=False)

    assert 2 == score_no_duplicate.nb
    assert 1.0 == score_no_duplicate.cosadd
    assert 0.5 == score_no_duplicate.cosmul

    # REPORTS
    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              2/3     100.00%      50.00%
                                                          (-1 duplicate)
------------------------------------------------------------------------------------------------
"""
    assert expected_summary_report == evaluation.get_report(keep_duplicates=False)

    expected_detail_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              2/3     100.00%      50.00%
                                                          (-1 duplicate)

a b c d                                                                            d           d
a b e f                                                                            f           c

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(keep_duplicates=False, show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_duplicates_in_different_datasets(matrix_type):
    embedding = get_representation(matrix_type)
    dataset1 = mangoes.evaluation.analogy.Dataset("dataset1", ['a b c d', 'a b e f'])
    dataset2 = mangoes.evaluation.analogy.Dataset("dataset2", ['a b c d', 'a b e f'])

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset1, dataset2)

    assert 0 == evaluation._questions_by_subset["dataset1"].nb_duplicates
    assert 0 == evaluation._questions_by_subset["dataset2"].nb_duplicates


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_with_subset_duplicates(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.analogy.Dataset("dataset", {"subset1": ['a b c d', 'a b e f'],
                                                             "subset2": ['c d a b', 'e f a b', 'a b c d']})

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset)

    assert 0 == evaluation._questions_by_subset["dataset/subset1"].nb_duplicates
    assert 0 == evaluation._questions_by_subset["dataset/subset2"].nb_duplicates
    assert 1 == evaluation._questions_by_subset["dataset"].nb_duplicates

    # SCORES
    score_with_duplicate = evaluation.get_score(keep_duplicates=True)
    score_no_duplicate = evaluation.get_score(keep_duplicates=False)

    assert 5 == score_with_duplicate.nb
    assert 4 == score_no_duplicate.nb

    assert 1.0 == score_with_duplicate.cosadd
    assert 0.8 == score_with_duplicate.cosmul

    assert 1.0 == score_no_duplicate.cosadd
    assert 0.75 == score_no_duplicate.cosmul


#########################
# Display report
# More tests to check reports

@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_with_subset_keep_duplicates_display_report(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.analogy.Dataset("dataset", {"subset1": ['a b c d', 'a b e f'],
                                                             "subset2": ['c d a b', 'e f a b', 'a b c d']})

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset)
    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              5/5     100.00%      80.00%
                                                 (including 1 duplicate)
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == evaluation.get_report()
    assert expected_summary_report == str(evaluation.get_report())

    # detail : with subsets
    expected_detail_report_subset = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              5/5     100.00%      80.00%
                                                 (including 1 duplicate)

    subset1                                                          2/2     100.00%      50.00%
    subset2                                                          3/3     100.00%     100.00%

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report_subset == evaluation.get_report(show_subsets=True)

    # detail : with subsets and predictions
    expected_detail_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              5/5     100.00%      80.00%
                                                 (including 1 duplicate)

    subset1                                                          2/2     100.00%      50.00%

    a b c d                                                                        d           d
    a b e f                                                                        f           c

    subset2                                                          3/3     100.00%     100.00%

    c d a b                                                                        b           b
    e f a b                                                                        b           b
    a b c d                                                                        d           d

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_with_subset_no_duplicates_display_report(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.analogy.Dataset("dataset", {"subset1": ['a b c d', 'a b e f', 'a b c d'],
                                                             "subset2": ['c d a b', 'e f a b', 'a b c d']})

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset)

    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              4/6     100.00%      75.00%
                                                         (-2 duplicates)
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == evaluation.get_report(keep_duplicates=False)
    assert expected_summary_report == str(evaluation.get_report(keep_duplicates=False))

    # detail : with subsets
    expected_detail_report_subset = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              4/6     100.00%      75.00%
                                                         (-2 duplicates)

    subset1                                                          2/3     100.00%      50.00%
                                                          (-1 duplicate)
    subset2                                                          3/3     100.00%     100.00%

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report_subset == evaluation.get_report(keep_duplicates=False, show_subsets=True)

    # detail : with subsets and predictions
    expected_detail_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              4/6     100.00%      75.00%
                                                         (-2 duplicates)

    subset1                                                          2/3     100.00%      50.00%
                                                          (-1 duplicate)

    a b c d                                                                        d           d
    a b e f                                                                        f           c

    subset2                                                          3/3     100.00%     100.00%

    c d a b                                                                        b           b
    e f a b                                                                        b           b
    a b c d                                                                        d           d

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(keep_duplicates=False, show_questions=True)


@pytest.mark.parametrize("matrix_type", [np.array, scipy.sparse.csr_matrix], ids=["dense", "sparse"])
def test_analogy_lower(matrix_type):
    embedding = get_representation(matrix_type)
    dataset = mangoes.evaluation.analogy.Dataset("dataset", ['A B C D', 'A b E f', ' a e b F', 'b f e c'])

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset, lower=True)

    # SCORES
    score = evaluation.get_score()
    assert 4 == score.nb
    assert 1.0 == score.cosadd
    assert 0.5 == score.cosmul

    # REPORTS
    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              4/4     100.00%      50.00%
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == evaluation.get_report()
    assert expected_summary_report == str(evaluation.get_report())

    expected_detail_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              4/4     100.00%      50.00%

a b c d                                                                            d           d
a b e f                                                                            f           c
a e b f                                                                            f           c
b f e c                                                                            c           c

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(show_questions=True)

    evaluation = mangoes.evaluation.analogy.Evaluation(embedding, dataset, lower=False)

    # SCORES
    score = evaluation.get_score()
    assert 1 == score.nb
    assert 1.0 == score.cosadd
    assert 1.0 == score.cosmul

    # REPORTS
    expected_summary_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              1/4     100.00%     100.00%
------------------------------------------------------------------------------------------------
"""

    assert expected_summary_report == evaluation.get_report()
    assert expected_summary_report == str(evaluation.get_report())

    expected_detail_report = """
                                                            Nb questions      cosadd      cosmul
================================================================================================
dataset                                                              1/4     100.00%     100.00%

b f e c                                                                            c           c

------------------------------------------------------------------------------------------------
"""
    assert expected_detail_report == evaluation.get_report(show_questions=True)
