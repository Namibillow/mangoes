# -*- coding: utf-8 -*-

import logging

import numpy as np
import pytest

import mangoes.utils.arrays
import mangoes.utils.metrics

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_rowwise_cosine_similarity():
    x = np.array([1, 0])
    y = np.array([0, 1])

    assert mangoes.utils.metrics.rowwise_cosine_similarity(x, x) == 1
    assert mangoes.utils.metrics.rowwise_cosine_similarity(x, y) == 0

    A = np.array([x, y, x, y])
    B = np.array([x, y, y, x])

    np.testing.assert_array_equal([1., 1., 0., 0.],
                                  mangoes.utils.metrics.rowwise_cosine_similarity(A, B))


@pytest.mark.unittest
def test_non_negative_cosine_similarity():
    x = mangoes.utils.arrays.Matrix.factory(np.array([[0, 1]]))
    y = mangoes.utils.arrays.Matrix.factory(np.array([[1, 0]]))

    assert 1 == mangoes.utils.metrics.pairwise_non_negative_cosine_similarity(x, x)
    assert 1 / 2 == mangoes.utils.metrics.pairwise_non_negative_cosine_similarity(x, y)

    x = [0, 1]
    y = [1, 0]
    A = mangoes.utils.arrays.Matrix.factory(np.array([x, y]))
    B = mangoes.utils.arrays.Matrix.factory(np.array([x, y, x, y]))

    np.testing.assert_array_equal(np.array([[1., 0.5, 1., 0.5],
                                            [0.5, 1., 0.5, 1.]]),
                                  mangoes.utils.metrics.pairwise_non_negative_cosine_similarity(A, B))


def test_earth_mover_distance_null():
    assert 0 == mangoes.utils.metrics._earth_mover_distance([0], [0], np.array([[0]]))[0]


def test_earth_mover_distance_one_move_equal_cost():
    d1 = [1, 0] # [x0, x1]
    d2 = [0, 1] # [y0, y1]
    cost = np.ones((2,2))

    # expected flow :
    # - x0 -> y1 -> cost = 1
    #                                y0  y1
    expected_flow_matrix = np.array([[0,  1],  # x0
                                     [0,  0]]) # x1

    distance, flow_matrix = mangoes.utils.metrics._earth_mover_distance(d1, d2, cost)

    assert 1 == distance
    np.testing.assert_array_equal(expected_flow_matrix, flow_matrix)


def test_earth_mover_distance_two_moves():
    d1 = [1, 1, 0] # [x0, x1, x2]
    d2 = [1, 0, 1] # [y0, y1, y2]
    cost = np.array([[0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 0]])

    # expected flow :
    # - x0 -> y0 -> cost = 0
    # - x1 -> y2 -> cost = 1
    #                                y0  y1  y2
    expected_flow_matrix = np.array([[1,  0,  0],  # x0
                                     [0,  0,  1],  # x1
                                     [0,  0,  0]]) # x2

    distance, flow_matrix = mangoes.utils.metrics._earth_mover_distance(d1, d2, cost)

    assert 1 == distance
    np.testing.assert_array_equal(expected_flow_matrix, flow_matrix)


def test_earth_mover_distance_three_moves():
    d1 = [1, 1, 1] # [x0, x1, x2]
    d2 = [1, 0, 2] # [y0, y1, y2]
    cost = np.array([[0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 0]])

    # expected flow :
    # - x0 -> y0 -> cost = 0
    # - x1 -> y2 -> cost = 1
    # - x2 -> y2 -> cost = 0
    #                                y0  y1  y2
    expected_flow_matrix = np.array([[1,  0,  0],  # x0
                                     [0,  0,  1],  # x1
                                     [0,  0,  1]]) # x2

    distance, flow_matrix = mangoes.utils.metrics._earth_mover_distance(d1, d2, cost)

    assert 1 == distance
    np.testing.assert_array_equal(expected_flow_matrix, flow_matrix)

    d1 = [1, 1, 1] # [x0, x1, x2]
    d2 = [1, 0, 2] # [y0, y1, y2]
    cost = np.array([[0, 2, 1],
                     [2, 0, 2],
                     [1, 2, 0]])

    # expected flow :
    # - x0 -> y0 -> cost = 0
    # - x1 -> y2 -> cost = 2
    # - x2 -> y2 -> cost = 0
    #                                y0  y1  y2
    expected_flow_matrix = np.array([[1,  0,  0],  # x0
                                     [0,  0,  1],  # x1
                                     [0,  0,  1]]) # x2

    distance, flow_matrix = mangoes.utils.metrics._earth_mover_distance(d1, d2, cost)

    assert 2 == distance
    np.testing.assert_array_equal(expected_flow_matrix, flow_matrix)


def test_earth_mover_distance_compare_to_pyemd():
    pyemd = pytest.importorskip("pyemd")
    d1 = np.array([1, 1, 1], dtype=float) # [x0, x1, x2]
    d2 = np.array([1, 0, 2], dtype=float) # [y0, y1, y2]
    cost = np.array([[0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 0]], dtype=float)

    d_pyemd, flow_pyemd = pyemd.emd_with_flow(d1, d2, cost)
    d_mangoes, flow_mangoes = mangoes.utils.metrics._earth_mover_distance(d1, d2, cost)

    np.testing.assert_almost_equal(d_pyemd, d_mangoes, decimal=4)
    np.testing.assert_array_almost_equal(flow_pyemd, flow_mangoes, decimal=4)


def test_earth_mover_distance_compare_to_pot():
    ot = pytest.importorskip("ot")
    d1 = np.array([1, 1, 1], dtype=float) # [x0, x1, x2]
    d2 = np.array([1, 0, 2], dtype=float) # [y0, y1, y2]
    cost = np.array([[0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 0]], dtype=float)

    d_pot, log = ot.emd2(d1, d2, cost, return_matrix=True)
    flow_pot = log['G']
    d_mangoes, flow_mangoes = mangoes.utils.metrics._earth_mover_distance(d1, d2, cost)

    np.testing.assert_almost_equal(d_pot, d_mangoes, decimal=4)
    np.testing.assert_array_almost_equal(flow_pot, flow_mangoes, decimal=4)


def test_word_mover_distance_one_words():
    representation = mangoes.Embeddings(mangoes.Vocabulary(['a', 'b']),
                                        mangoes.utils.arrays.Matrix.factory(np.array([[1, 0],
                                                                                      [1, 1]])))

    distance = mangoes.utils.metrics.word_mover_distance(representation, "a", "b")
    assert 1 == distance

    distance, flow = mangoes.utils.metrics.word_mover_distance(representation, "a", "b", return_flow=True)
    assert 1 == distance
    assert {"a": {("b", 1)}} == flow


def test_word_mover_distance_twice_the_same_word_move_to_two_different_words():
    representation = mangoes.Embeddings(mangoes.Vocabulary(['a', 'b', 'c']),
                                        mangoes.utils.arrays.Matrix.factory(np.array([[1, 0],
                                                                                      [1, 1],
                                                                                      [2, 0]])))

    distance = mangoes.utils.metrics.word_mover_distance(representation, "a a", "b c")
    np.testing.assert_almost_equal(1, distance, decimal=4)

    distance, flow = mangoes.utils.metrics.word_mover_distance(representation, "a a", "b c", return_flow=True)
    np.testing.assert_almost_equal(1, distance, decimal=4)
    assert {"a": {("b", 0.5), ("c", 0.5)}} == flow


def test_word_mover_distance_different_numbers_of_words():
    representation = mangoes.Embeddings(mangoes.Vocabulary(['a', 'b', 'c']),
                                        mangoes.utils.arrays.Matrix.factory(np.array([[1, 0],
                                                                                      [1, 1],
                                                                                      [2, 0]])))

    distance = mangoes.utils.metrics.word_mover_distance(representation, "a a", "b b c")
    np.testing.assert_almost_equal(1, distance, decimal=4)

    distance, flow = mangoes.utils.metrics.word_mover_distance(representation, "a a", "b c", return_flow=True)
    np.testing.assert_almost_equal(1, distance, decimal=4)
    assert {"a": {("b", 0.5), ("c", 0.5)}} == flow


def test_word_mover_distance_symmetry():
    representation = mangoes.Embeddings(mangoes.Vocabulary(['a', 'b', 'c']),
                                        mangoes.utils.arrays.Matrix.factory(np.random.rand(3, 10)))

    s1 = "a a b c a b c"
    s2 = "b b c a c b a c a b c"

    distance1 = mangoes.utils.metrics.word_mover_distance(representation, s1, s2)
    distance2 = mangoes.utils.metrics.word_mover_distance(representation, s2, s1)

    np.testing.assert_almost_equal(distance1, distance2)


def test_word_mover_distance_obama():
    representation = mangoes.Embeddings(mangoes.Vocabulary(['obama', 'speaks', 'media', 'illinois',
                                                            'president', 'greets', 'press', 'chicago']),
                                        mangoes.utils.arrays.Matrix.factory(np.array([[1, 0, 0, 0],
                                                                                      [0, 1, 0, 0],
                                                                                      [0, 0, 1, 0],
                                                                                      [0, 0, 0, 1],
                                                                                      [1.45, 0, 0, 0],
                                                                                      [0, 1.24, 0, 0],
                                                                                      [0, 0, 1.2, 0],
                                                                                      [0, 0, 0, 1.18]])))

    distance, flow = mangoes.utils.metrics.word_mover_distance(representation,
                                                               "obama speaks to the media in illinois",
                                                               "the president greets the press in chicago",
                                                               stopwords=['to', 'the', 'in'], return_flow=True,
                                                               emd=mangoes.utils.metrics._earth_mover_distance)
    np.testing.assert_almost_equal(1.07/4, distance, decimal=4) # TODO : should the distance be 1.07 or 1.07/4 ?

    assert {("president", 0.25)} == flow['obama']
    assert {("greets", 0.25)} == flow['speaks']
    assert {("press", 0.25)} == flow['media']
    assert {("chicago", 0.25)} == flow['illinois']


def test_word_mover_distance_pot():
    pytest.importorskip('ot')
    representation = mangoes.Embeddings(mangoes.Vocabulary(['obama', 'speaks', 'media', 'illinois',
                                                            'president', 'greets', 'press', 'chicago']),
                                        mangoes.utils.arrays.Matrix.factory(np.array([[1, 0, 0, 0],
                                                                                      [0, 1, 0, 0],
                                                                                      [0, 0, 1, 0],
                                                                                      [0, 0, 0, 1],
                                                                                      [1.45, 0, 0, 0],
                                                                                      [0, 1.24, 0, 0],
                                                                                      [0, 0, 1.2, 0],
                                                                                      [0, 0, 0, 1.18]])))

    distance, flow = mangoes.utils.metrics.word_mover_distance(representation,
                                                               "obama speaks to the media in illinois",
                                                               "the president greets the press in chicago",
                                                               stopwords=['to', 'the', 'in'],
                                                               return_flow=True,
                                                               emd="pot")
    np.testing.assert_almost_equal(1.07/4, distance, decimal=4) # TODO : should the distance be 1.07 or 1.07/4 ?

    assert {("president", 0.25)} == flow['obama']
    assert {("greets", 0.25)} == flow['speaks']
    assert {("press", 0.25)} == flow['media']
    assert {("chicago", 0.25)} == flow['illinois']


def test_word_mover_distance_pyemd():
    pytest.importorskip('pyemd')
    representation = mangoes.Embeddings(mangoes.Vocabulary(['obama', 'speaks', 'media', 'illinois',
                                                            'president', 'greets', 'press', 'chicago']),
                                        mangoes.utils.arrays.Matrix.factory(np.array([[1, 0, 0, 0],
                                                                                      [0, 1, 0, 0],
                                                                                      [0, 0, 1, 0],
                                                                                      [0, 0, 0, 1],
                                                                                      [1.45, 0, 0, 0],
                                                                                      [0, 1.24, 0, 0],
                                                                                      [0, 0, 1.2, 0],
                                                                                      [0, 0, 0, 1.18]])))

    distance, flow = mangoes.utils.metrics.word_mover_distance(representation,
                                                               "obama speaks to the media in illinois",
                                                               "the president greets the press in chicago",
                                                               stopwords=['to', 'the', 'in'],
                                                               return_flow=True,
                                                               emd="pyemd")
    np.testing.assert_almost_equal(1.07/4, distance, decimal=4) # TODO : should the distance be 1.07 or 1.07/4 ?

    assert {("president", 0.25)} == flow['obama']
    assert {("greets", 0.25)} == flow['speaks']
    assert {("press", 0.25)} == flow['media']
    assert {("chicago", 0.25)} == flow['illinois']


@pytest.mark.skip("Keep this as a reminder that local implementation is likely to fail for large sentences.")
def test_wmd_large_documents():
    import string
    vocabulary = mangoes.Vocabulary(list(string.ascii_letters))
    matrix = np.random.rand(len(vocabulary), 50)

    representation = mangoes.Embeddings(vocabulary, matrix)

    s1 = [vocabulary[i] for i in np.random.randint(0, len(vocabulary), 200)]
    s2 = [vocabulary[i] for i in np.random.randint(0, len(vocabulary), 200)]

    with pytest.raises(RuntimeError):
        mangoes.utils.metrics.word_mover_distance(representation, s1, s2, return_flow=True,
                                                  emd = mangoes.utils.metrics._earth_mover_distance)

    mangoes.utils.metrics.word_mover_distance(representation, s1, s2, return_flow=True)


def test_wmd_oov():
    representation = mangoes.Embeddings(mangoes.Vocabulary(['a', 'b', 'c']),
                                        mangoes.utils.arrays.Matrix.factory(np.random.rand(3, 10)))

    s1 = "a b c"
    s2 = "d e f"

    assert np.isnan(mangoes.utils.metrics.word_mover_distance(representation, s1, s2))
