# -*- coding: utf-8 -*-

import logging

import numpy as np
import pytest

import mangoes
import mangoes.composition

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


class FakeRepresentation(mangoes.base.Representation):
    def to_df(self):
        pass


@pytest.mark.parametrize("matrix_type", [np.array], ids=["dense"])
def test_compose_multiplicative(matrix_type):
    words = ['x', 'y']
    x, y = [1, 2, 3], [4, 5, 6]
    vectors = matrix_type([x, y])
    representation = FakeRepresentation(mangoes.Vocabulary(words),
                                        vectors)

    m = mangoes.composition.MultiplicativeComposer(representation)
    np.testing.assert_array_equal([4, 10, 18], m.predict('x', 'y'))


@pytest.mark.parametrize("matrix_type", [np.array], ids=["dense"])
def test_learn_weighted_additive_parameters(matrix_type):
    words = ['x', 'y', 'z', 'x y', 'x z']
    x, y, z = [0, 1], [1, 0], [1, 1]

    # a = 0.4, b = 0.6

    # ax + by = [0.6, 0.4]
    # ax + bz = [0.6, 1]
    vectors = matrix_type([x, y, z, [0.6, 0.4], [0.6, 1]])
    representation = FakeRepresentation(mangoes.Vocabulary(words),
                                        vectors)

    composer = mangoes.composition.AdditiveComposer(representation)
    composer.fit()

    np.testing.assert_almost_equal(0.4, composer.alpha)
    np.testing.assert_almost_equal(0.6, composer.beta)


@pytest.mark.parametrize("matrix_type", [np.array], ids=["dense"])
def test_learn_dilation_parameter(matrix_type):
    words = ['x', 'y', 'z', 'x y', 'x z']
    x, y, z = [0, 1], [1, 0], [1, 1]

    # lambda = 1.69

    # x.x = 1 / x.y = 0 / x.z = 1
    # (l - 1)(x.y)x + (x.x)y = y = [1, 0]
    # (l - 1)(x.z)x + (x.x)z = x + z = [1, 1.69]
    vectors = matrix_type([x, y, z, [1, 0], [1, 1.69]])
    representation = FakeRepresentation(mangoes.Vocabulary(words),
                                        vectors)

    composer = mangoes.composition.DilationComposer(representation)
    composer.fit()
    np.testing.assert_almost_equal(1.69, composer.lambda_)


@pytest.mark.parametrize("matrix_type", [np.array], ids=["dense"])
def test_learn_full_additive_parameters_random(matrix_type):
    words = ['a', 'b', 'c', 'd', 'e',
             'a a', 'a b', 'a c', 'a d', 'a e',
             'b a', 'b b', 'b c', 'b d', 'b e',
             'c a', 'c b', 'c c', 'c d', 'c e',
             'd a', 'd b', 'd c', 'd d', 'd e',
             'e a', 'e b', 'e c', 'e d', 'e e']

    a, b, c, d, e = matrix_type([0, 1]), matrix_type([1, 0]), matrix_type([1, 1]), matrix_type([0, 2]), matrix_type([2, 1])

    expected_A = matrix_type(np.random.rand(2, 2))
    expected_B = matrix_type(np.random.rand(2, 2))

    vectors = np.zeros((30, 2))
    vectors[0:5] = [a, b, c, d, e]
    for i, (u, v) in enumerate([(a, a), (a, b), (a, c), (a, d), (a, e),
                                (b, a), (b, b), (b, c), (b, d), (b, e),
                                (c, a), (c, b), (c, c), (c, d), (c, e),
                                (d, a), (d, b), (d, c), (d, d), (d, e),
                                (e, a), (e, b), (e, c), (e, d), (e, e)], start=5):
        vectors[i] = expected_A.dot(u) + expected_B.dot(v)

    representation = FakeRepresentation(mangoes.Vocabulary(words),
                                        matrix_type(vectors))

    composer = mangoes.composition.FullAdditiveComposer(representation)
    composer.fit()

    np.testing.assert_array_almost_equal(representation[('a', 'b')], composer.predict('a', 'b'))

    np.testing.assert_array_almost_equal(expected_A, composer.A)
    np.testing.assert_array_almost_equal(expected_B, composer.B)


@pytest.mark.parametrize("matrix_type", [np.array], ids=["dense"])
def test_learn_lexical_id(matrix_type):
    words = ['a', 'b', 'c', 'x a', 'x b', 'x c']
    a, b, c = [1, 1], [1, 0], [0, 1]

    # X = Id
    vectors = matrix_type(np.array([a, b, c, a, b, c]))
    representation = FakeRepresentation(mangoes.Vocabulary(words),
                                        vectors)

    x_composer = mangoes.composition.LexicalComposer(representation, 'x')
    x_composer.fit()

    np.testing.assert_array_almost_equal([1, 1], x_composer.predict('a'))
    np.testing.assert_array_almost_equal([1, 0], x_composer.predict('b'))

    np.testing.assert_array_almost_equal(np.eye(2), x_composer.U)


@pytest.mark.parametrize("matrix_type", [np.array],
                         ids=["dense"])  # [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_learn_lexical_random(matrix_type):
    words = ['a', 'b', 'c', 'x a', 'x b', 'x c']
    a, b, c = [1, 1], [1, 0], [0, 1]

    # X = random
    X = np.random.rand(2,2)
    vectors = matrix_type(np.array([a, b, c, X.dot(a), X.dot(b), X.dot(c)]))
    representation = FakeRepresentation(mangoes.Vocabulary(words),
                                        vectors)

    x_composer = mangoes.composition.LexicalComposer(representation, 'x')
    x_composer.fit()

    np.testing.assert_array_almost_equal(representation[('x', 'a')], x_composer.predict('a'))
    np.testing.assert_array_almost_equal(representation[('x', 'b')], x_composer.predict('b'))

    np.testing.assert_array_almost_equal(X, x_composer.U)
