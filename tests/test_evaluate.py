# -*- coding: utf-8 -*-

import logging
from unittest import mock
import pytest

import numpy as np
import scipy.sparse

import mangoes.base
import mangoes.dataset
import mangoes.evaluate
import mangoes.utils.arrays

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


# ########


# ################# Analogy
def test_analogy_on_an_empty_embedding_is_NA():
    embeddings = mangoes.base.Embeddings(mangoes.Vocabulary({}), np.ndarray([]))
    dataset = mangoes.dataset.Dataset([])

    result = mangoes.evaluate.analogy(embeddings, dataset)

    assert mangoes.evaluate.NA == result


def test_analogy():
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e', 'f'])
    matrix = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [-1, 1, 1],
                       [2, 1, 1],
                       [1, 2, 1]])
    embedding = mangoes.base.Embeddings(words, matrix)
    dataset = mangoes.dataset.Dataset(['a b c d', 'a b e f'], name="dataset")

    result = mangoes.evaluate.analogy(embedding, dataset=dataset)

    assert 1.0 == result.score.cosadd
    assert 0.5 == result.score.cosmul
    assert set() == result.oov
    assert 0 == result.ignored
    assert {"dataset": {"score": (1.0, 0.5),
                        "nb_questions_in_subset": 2,
                        "questions": {"a b c d": (['d'], ['d']),
                                      "a b e f": (['f'], ['d'])}}} == result._results_dict

    print("\n" + result.more_detail)


def test_analogy_with_a_sparse_embedding():
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e', 'f'])
    matrix = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [-1, 1, 1],
                       [2, 1, 1],
                       [1, 2, 1]])
    embedding = mangoes.base.Embeddings(words, scipy.sparse.csr_matrix(matrix))
    dataset = mangoes.dataset.Dataset(['a b c d', 'a b e f'])

    result = mangoes.evaluate.analogy(embedding, dataset)

    assert 1 == result.score.cosadd
    assert 0 == result.ignored
    assert {"": {"score": 1,
                 "questions": {"a b c d": 'd', "a b e f": 'f'}}}


def test_analogy_on_a_dataset_with_subsets():
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e', 'f'])
    matrix = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [-1, 1, 1],
                       [2, 1, 1],
                       [1, 2, 1]])
    embedding = mangoes.base.Embeddings(words, matrix)

    dataset = mangoes.dataset.Dataset({"subset1": ['a b c d', 'a b e f'],
                                      "subset2": ['a b c f', 'a b e d']}, name="dataset")

    result = mangoes.evaluate.analogy(embedding, dataset=dataset)

    assert (0.5, 0.5) == result.score
    assert 0 == result.ignored
    assert set() == result.oov

    assert (1.0, 0.5) == result.get_score("/dataset/subset1")
    assert (0.0, 0.5) == result.get_score("/dataset/subset2")


def test_analogy_on_a_dataset_with_words_out_of_vocabulary():
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e', 'f'])
    matrix = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [-1, 1, 1],
                       [2, 1, 1],
                       [1, 2, 1]])
    embedding = mangoes.base.Embeddings(words, matrix)

    dataset = mangoes.dataset.Dataset(['a b c d',
                                      'a b e z',
                                      'a b z e'], name="dataset")

    result = mangoes.evaluate.analogy(embedding, dataset=dataset)

    assert (1.0, 1.0) == result.score
    assert 2 == result.ignored
    assert {'z'} == result.oov

    assert result.detail


def test_analogy_with_2_answers_allowed():
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e', 'f'])
    matrix = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [-1, 1, 1],
                       [2, 1, 1],
                       [1, 2, 1]])
    embedding = mangoes.base.Embeddings(words, matrix)
    dataset = mangoes.Dataset(['a b c d', 'a b e f'])

    result = mangoes.evaluate.analogy(embedding, dataset, allowed_answers=3)

    assert (1, 1) == result.score
    assert {"": {"score": (1.0, 1.0),
                 "nb_questions_in_subset": 2,
                 "questions": {"a b c d": (['d', 'f', 'e'], ['d', 'f', 'e']),
                               "a b e f": (['f', 'd', 'c'], ['d', 'f', 'c'])}}} == result._results_dict


# ################# Similarity
def test_similarity_on_an_empty_embedding_is_NA():
    embeddings = mangoes.base.Embeddings(mangoes.Vocabulary({}), np.ndarray([]))
    dataset = mangoes.dataset.Dataset([])

    result = mangoes.evaluate.similarity(embeddings, dataset)

    assert mangoes.evaluate.NA == result


def test_similarity():
    words = mangoes.Vocabulary(['a', 'b', 'c'])
    matrix = np.array([[1, 0, 0],
                       [1, 0, 1],
                       [1, 2, 0]])
    embedding = mangoes.base.Embeddings(words, matrix)
    dataset = mangoes.dataset.Dataset(['a b 0.5', 'a c 0.3', 'b c 0.4'])

    result = mangoes.evaluate.similarity(embedding, dataset)

    assert -1 < result.score.pearson.coeff < 1
    assert 0.05 < result.score.pearson.pvalue
    assert -1 < result.score.spearman.coeff < 1
    assert 0.05 < result.score.spearman.pvalue
    assert set() == result.oov
    assert 0 == result.ignored

    assert 3 == len(result._results_dict[""]["questions"])


def test_similarity_with_a_sparse_embedding():
    words = mangoes.Vocabulary(['a', 'b', 'c'])
    matrix = np.array([[1, 0, 0],
                       [1, 0, 1],
                       [1, 2, 0]])
    embedding = mangoes.base.Embeddings(words, scipy.sparse.csr_matrix(matrix))
    dataset = mangoes.Dataset(['a b 0.5', 'a c 0.3', 'b c 0.4'])

    result = mangoes.evaluate.similarity(embedding, dataset)

    assert -1 < result.score.pearson.coeff < 1
    assert 0.05 < result.score.pearson.pvalue
    assert -1 < result.score.spearman.coeff < 1
    assert 0.05 < result.score.spearman.pvalue
    assert 0 == result.ignored


def test_similarity_on_a_dataset_with_subsets():
    words = mangoes.Vocabulary(['a', 'b', 'c'])
    matrix = np.array([[1, 0, 0],
                       [1, 0, 1],
                       [1, 2, 0]])
    embedding = mangoes.base.Embeddings(words, matrix)
    dataset = mangoes.dataset.Dataset({"subset1": ['a b 0.5', 'a c 0.3', 'b c 0.4'],
                                      "subset2": ['a b 0.5', 'a c 0.3', 'b c 0.4']})

    result = mangoes.evaluate.similarity(embedding, dataset)

    assert -1 < result.score.pearson.coeff < 1
    assert 0.05 < result.score.pearson.pvalue
    assert -1 < result.score.spearman.coeff < 1
    assert 0.05 < result.score.spearman.pvalue
    assert set() == result.oov
    assert 0 == result.ignored

    assert -1 < result._results_dict[""]["subsets"]["subset1"]["score"].pearson.coeff < 1
    assert -1 < result._results_dict[""]["subsets"]["subset2"]["score"].pearson.coeff < 1


def test_similarity_on_a_dataset_with_words_out_of_vocabulary():
    words = mangoes.Vocabulary(['a', 'b', 'c'])
    matrix = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
    embedding = mangoes.base.Embeddings(words, matrix)
    dataset = mangoes.dataset.Dataset(['a b 0.5', 'a c 0.3', 'b z 0.4'])

    result = mangoes.evaluate.similarity(embedding, dataset)

    assert 1 == result.ignored
    assert {'z'} == result.oov
    assert result.detail


# ################# Outlier Detection
def test_outlier_detection_on_an_empty_embedding_is_NA():
    embeddings = mangoes.base.Embeddings(mangoes.Vocabulary({}), np.ndarray([]))
    dataset = mangoes.dataset.Dataset([])

    result = mangoes.evaluate.outlier_detection(embeddings, dataset)

    assert mangoes.evaluate.NA == result


def test_outlier_detection():
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e'])
    matrix = np.array([[1, 0, 0],
                       [0.9, 0, 0],
                       [0.8, 0, 0],
                       [1.1, 0, 0],
                       [0, 1, 0]])
    embedding = mangoes.base.Embeddings(words, matrix)
    dataset = mangoes.dataset.OutlierDetectionDataset(['a b c d e'])

    result = mangoes.evaluate.outlier_detection(embedding, dataset)

    assert (1, 1) == result.score
    assert set() == result.oov
    assert 0 == result.ignored

    assert 1 == len(result._results_dict[""]["questions"])


def test_outlier_detection_with_a_sparse_embedding():
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e'])
    matrix = np.array([[1, 0, 0],
                       [0.9, 0, 0],
                       [0.8, 0, 0],
                       [1.1, 0, 0],
                       [0, 1, 0]])
    embedding = mangoes.base.Embeddings(words, scipy.sparse.csr_matrix(matrix))
    dataset = mangoes.dataset.OutlierDetectionDataset(['a b c d e'])

    result = mangoes.evaluate.outlier_detection(embedding, dataset)

    assert (1, 1) == result.score
    assert set() == result.oov
    assert 0 == result.ignored

    assert 1 == len(result._results_dict[""]["questions"])


def test_outlier_detection_on_a_dataset_with_subsets():
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e'])
    matrix = np.array([[1, 0, 0],
                       [0.9, 0, 0],
                       [0.8, 0, 0],
                       [1.1, 0, 0],
                       [0, 1, 0]])
    embedding = mangoes.base.Embeddings(words, matrix)
    dataset = mangoes.dataset.OutlierDetectionDataset({"subset1": ['a b c d e'],
                                                      "subset2": ['a b c e d']})

    result = mangoes.evaluate.outlier_detection(embedding, dataset)

    assert 0.5 == result.score.accuracy
    assert set() == result.oov
    assert 0 == result.ignored

    assert (1, 1) == result._results_dict[""]["subsets"]["subset1"]["score"]


def test_outlier_detection_on_a_dataset_with_words_out_of_vocabulary():
    words = mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e'])
    matrix = np.array([[1, 0, 0],
                       [0.9, 0, 0],
                       [0.8, 0, 0],
                       [1.1, 0, 0],
                       [0, 1, 0]])
    embedding = mangoes.base.Embeddings(words, matrix)
    dataset = mangoes.dataset.OutlierDetectionDataset(['a b c d e', 'a b c d z'])

    result = mangoes.evaluate.outlier_detection(embedding, dataset)

    assert 1 == result.ignored
    assert {'z'} == result.oov
    assert result.detail


# ################# Distances

class StubEmbeddings:
    def __init__(self):
        self.matrix = mangoes.utils.arrays.Matrix.factory(np.array([[0, 1],
                                                                    [1, 0],
                                                                    [1, 1],
                                                                    [0, 1]]))
        self.words = mangoes.Vocabulary(["a", "b", "c", "d"])


@pytest.mark.unittest
def test_distances_from_one_word_histogram():
    embeddings = StubEmbeddings()

    theta = np.linspace(0.0, np.pi, 4, endpoint=True)  # [0, pi/3, 2pi/3, pi] => 3 intervals

    np.testing.assert_equal([2, 1, 0], mangoes.evaluate.distances_one_word_histogram(embeddings, "a", theta))
    np.testing.assert_equal([1, 2, 0], mangoes.evaluate.distances_one_word_histogram(embeddings, "b", theta))
    np.testing.assert_equal([3, 0, 0], mangoes.evaluate.distances_one_word_histogram(embeddings, "c", theta))
    np.testing.assert_equal([2, 1, 0], mangoes.evaluate.distances_one_word_histogram(embeddings, "d", theta))


@pytest.mark.unittest
def test_distances_between_words_histogram():
    embeddings = StubEmbeddings()

    theta = np.linspace(0.0, np.pi, 4, endpoint=True)  # [0, pi/3, 2pi/3, pi] => 3 intervals

    np.testing.assert_equal([4, 2, 0], mangoes.evaluate.distances_histogram(embeddings, theta))
