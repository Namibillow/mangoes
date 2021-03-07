import numpy as np

import pytest

import mangoes


@pytest.mark.unittest
def test_angles():
    matrix = mangoes.utils.arrays.Matrix.factory(np.array([[0, 1],
                                                           [1, 0],
                                                           [1, 1],
                                                           [0, 1]]))

    vector = matrix[0]
    angles = mangoes.evaluation.statistics._angles(vector, matrix)
    expected = [0, np.pi / 2, np.pi / 4, 0]
    np.testing.assert_allclose(expected, angles)

    vector = matrix[1]
    angles = mangoes.evaluation.statistics._angles(vector, matrix)
    expected = [np.pi / 2, 0, np.pi / 4, np.pi / 2]
    np.testing.assert_allclose(expected, angles)


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

    np.testing.assert_equal([2, 1, 0],
                            mangoes.evaluation.statistics.distances_one_word_histogram(embeddings, "a", theta))
    np.testing.assert_equal([1, 2, 0],
                            mangoes.evaluation.statistics.distances_one_word_histogram(embeddings, "b", theta))
    np.testing.assert_equal([3, 0, 0],
                            mangoes.evaluation.statistics.distances_one_word_histogram(embeddings, "c", theta))
    np.testing.assert_equal([2, 1, 0],
                            mangoes.evaluation.statistics.distances_one_word_histogram(embeddings, "d", theta))


@pytest.mark.unittest
def test_distances_between_words_histogram():
    embeddings = StubEmbeddings()

    theta = np.linspace(0.0, np.pi, 4, endpoint=True)  # [0, pi/3, 2pi/3, pi] => 3 intervals

    np.testing.assert_equal([4, 2, 0], mangoes.evaluation.statistics.distances_histogram(embeddings, theta))
