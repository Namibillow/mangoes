# -*- coding: utf-8 -*-

import logging
import os
import pickle
from unittest import mock

import numpy as np
import pytest
from scipy import sparse

import mangoes
import mangoes.counting
import mangoes.utils.arrays
import mangoes.utils.exceptions

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


# ###########################################################################################
# ### Mocks
class DummyVocabulary:
    def __init__(self, words=[]):
        self.words = words
        self.params = {}
        self.index = self.words.index

    def __getitem__(self, i):
        return self.words[i]


class DummyCooccurrenceCount:
    def __init__(self):
        self.words = DummyVocabulary()
        self.contexts_words = DummyVocabulary()
        self.matrix = np.array([0])
        self.params = {}


class FakeRepresentation(mangoes.base.Representation):
    def to_df(self):
        pass


def dummy_factory(value):
    return value


# ###########################################################################################
# ### Unit tests

# create_representation function
@pytest.mark.unittest
def test_create_representation_default():
    with mock.patch.object(mangoes.utils.arrays.Matrix, 'factory', side_effect=dummy_factory):
        source = DummyCooccurrenceCount()
        result = mangoes.create_representation(source)
        np.testing.assert_array_equal(source.matrix, result.matrix)


@pytest.mark.unittest
def test_create_representation_with_weighting():
    mock_transformation = mock.Mock()
    mock_transformation.params = {"name": "mock"}
    mock_transformation.return_value = np.array([])

    with mock.patch.object(mangoes.utils.arrays.Matrix, 'factory', side_effect=dummy_factory):
        source = DummyCooccurrenceCount()
        result = mangoes.create_representation(source, weighting=mock_transformation)

        assert mock_transformation.called
        assert 1 == mock_transformation.call_count
        assert {"name": "mock"} == result._params["weighting"]


@pytest.mark.unittest
def test_create_representation_with_weighting_and_reduction():
    mock_transformation = mock.Mock()
    mock_transformation.params = {"name": "mock"}
    mock_transformation.return_value = np.array([])

    with mock.patch.object(mangoes.utils.arrays.Matrix, 'factory', side_effect=dummy_factory):
        source = DummyCooccurrenceCount()
        result = mangoes.create_representation(source,
                                               weighting=mock_transformation,
                                               reduction=mock_transformation)

        assert mock_transformation.called
        assert 2 == mock_transformation.call_count
    assert {"name": "mock"} == result._params["weighting"]
    assert {"name": "mock"} == result._params["reduction"]


@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
@pytest.mark.parametrize("metric, expected", [("cosine", [1, 1 - np.cos(np.pi/4)]),
                                              ("euclidean", [np.sqrt(2), 1]),
                                              ("cityblock", [2, 1]),
                                              ("chebyshev", [1, 1]),
                                              ])
def test_distance(matrix_type, metric, expected):
    class FakeRepresentation(mangoes.base.Representation):
        def to_df(self):
            pass

    words = mangoes.Vocabulary(["a", "b", "c"])
    matrix = matrix_type([[0, 1], [1, 0], [1, 1]])
    representation = FakeRepresentation(words, mangoes.utils.arrays.Matrix.factory(matrix))

    assert 0 == representation.distance("a", "a", metric=metric)

    assert expected[0] == representation.distance("a", "b", metric=metric)
    np.testing.assert_almost_equal(expected[1], representation.distance("a", "c", metric=metric))


@pytest.mark.parametrize("matrix_type", [np.array], ids=["dense"])
def test_distance_with_param(matrix_type):
    class FakeRepresentation(mangoes.base.Representation):
        def to_df(self):
            pass

    words = mangoes.Vocabulary(["a", "b", "c"])
    matrix = matrix_type([[0, 1], [1, 0], [1, 1]])
    representation = FakeRepresentation(words, mangoes.utils.arrays.Matrix.factory(matrix))

    assert 1 == representation.distance("a", "c", metric="minkowski", p=1)


@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
#                                                               ["a", "b"]     ["a", "b"], ["c"]
@pytest.mark.parametrize("metric, expected", [("cosine",    [ [[0,1], [1,0]], [[1 - np.cos(np.pi/4)],
                                                                               [1 - np.cos(np.pi/4)]]]),
                                              ("euclidean", [ [[0, np.sqrt(2)],  [np.sqrt(2), 0]], [[1], [1]]]),
                                              ("cityblock", [ [[0, 2], [2,0]],  [[1], [1]]]),
                                              ("chebyshev", [ [[0, 1], [1, 0]], [[1], [1]]]),
                                              ])
def test_pairwise_distances(matrix_type, metric, expected):
    class FakeRepresentation(mangoes.base.Representation):
        def to_df(self):
            pass

    words = mangoes.Vocabulary(["a", "b", "c"])
    matrix = matrix_type([[0, 1], [1, 0], [1, 1]])
    representation = FakeRepresentation(words, mangoes.utils.arrays.Matrix.factory(matrix))

    np.testing.assert_array_equal([[0]], representation.pairwise_distances(["a"], metric=metric))
    np.testing.assert_array_equal(expected[0], representation.pairwise_distances(["a", "b"], metric=metric))
    np.testing.assert_array_almost_equal(expected[1], representation.pairwise_distances(["a", "b"], ["c"],
                                                                                        metric=metric))


@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_pairwise_distances_3_2(matrix_type):
    class FakeRepresentation(mangoes.base.Representation):
        def to_df(self):
            pass

    words = mangoes.Vocabulary(["a", "b", "c"])
    matrix = matrix_type([[0, 1], [1, 0], [1, 1]])
    representation = FakeRepresentation(words, mangoes.utils.arrays.Matrix.factory(matrix))

    #                 a             b
    expected = [[0,             np.sqrt(2)],  # a
                [np.sqrt(2),            0],   # b
                [1,                     1]]   # c
    np.testing.assert_array_equal(expected,
                                  representation.pairwise_distances(["a", "b", "c"], ["a", "b"], metric='euclidean'))


# ###########################################################################################
# ### Integration tests

# ## Closest words
@pytest.mark.unitest
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_get_closest_words(matrix_type):
    matrix = [[0, 0, 1], [0, 0, 2], [0, 1, 0]]
    vocabulary = DummyVocabulary(['a', 'b', 'c'])
    representation = FakeRepresentation(vocabulary, matrix_type(matrix))
    result = representation.get_closest_words("a")

    assert result == [('b', 0), ('c', 1)]


@pytest.mark.unitest
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_get_closest_words_with_tokens(matrix_type):
    import collections
    Token = collections.namedtuple('Token', 'lemma POS')

    matrix = [[0, 0, 1], [0, 0, 2], [0, 1, 0]]
    vocabulary = DummyVocabulary([Token('a', 'A'), Token('b', 'B'), Token('c', 'C')])
    representation = FakeRepresentation(vocabulary, matrix_type(matrix))
    result = representation.get_closest_words(('a', 'A'))

    assert result == [(('b', 'B'), 0), (('c', 'C'), 1)]


@pytest.mark.unitest
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_get_closest_words_other_metrics(matrix_type):
    matrix = [[1, 0], [3, 2], [4, 0]]
    embeddings = FakeRepresentation(DummyVocabulary(['a', 'b', 'c']), matrix_type(matrix))

    assert ('b', 2*np.sqrt(2)) == embeddings.get_closest_words("a", nb=1, metric="euclidean")[0]
    assert ('c', 3) == embeddings.get_closest_words("a", nb=1, metric="manhattan")[0]


@pytest.mark.unitest
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_get_closest_words_of_a_vector(matrix_type):
    matrix = [[1, 0], [0.5, np.sqrt(3)/2], [0, 1]]
    embeddings = FakeRepresentation(DummyVocabulary(['a', 'b', 'c']), matrix_type(matrix))

    assert [('a', 0)] == embeddings.get_closest_words([1, 0], nb=1)
    assert [('c', 1)] == embeddings.get_closest_words([0, 2], nb=1, metric="manhattan")


@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_exception_oov_in_closest_words(matrix_type):
    with pytest.raises(mangoes.utils.exceptions.OutOfVocabulary):
        embeddings = FakeRepresentation(mangoes.Vocabulary([]), matrix_type([]))
        embeddings.get_closest_words("XXX")


# Persistence
@pytest.mark.integration
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_save_load_folder_count_based_representation(save_temp_dir, matrix_type):
    path = os.path.join(str(save_temp_dir), "test_save")

    words_vocabulary = mangoes.Vocabulary(["a", "b", "c"])
    contexts_vocabulary = mangoes.Vocabulary(["d", "e"])
    matrix = matrix_type([[0, 5], [3, 0], [6, 9]])
    counts = mangoes.CountBasedRepresentation(words_vocabulary, contexts_vocabulary, matrix, hyperparameters={"a": 0})

    # Expected results
    expected = mangoes.CountBasedRepresentation(mangoes.Vocabulary(["a", "b", "c"]),
                                                mangoes.Vocabulary(["d", "e"]),
                                                matrix_type([[0, 5], [3, 0], [6, 9]]))

    # Actual results
    counts.save(path)
    actual = mangoes.CountBasedRepresentation.load(path)

    # Test
    assert os.path.isdir(path)
    assert expected.words.words == actual.words.words
    assert expected.contexts_words.words == actual.contexts_words.words
    try:
        assert np.allclose(expected.matrix, actual.matrix)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected.matrix, actual.matrix)
    assert 0 == actual.params["a"]


@pytest.mark.integration
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_save_load_archive_count_based_representation(save_temp_dir, matrix_type):
    path = os.path.join(str(save_temp_dir), "test_save_archive.zip")

    words_vocabulary = mangoes.Vocabulary(["a", "b", "c"])
    contexts_vocabulary = mangoes.Vocabulary(["d", "e"])
    matrix = matrix_type([[0, 5], [3, 0], [6, 9]])
    cooc_count = mangoes.CountBasedRepresentation(words_vocabulary, contexts_vocabulary, matrix,
                                                  hyperparameters={"a": "x"})

    # Expected results
    expected = mangoes.CountBasedRepresentation(mangoes.Vocabulary(["a", "b", "c"]),
                                                mangoes.Vocabulary(["d", "e"]),
                                                matrix_type([[0, 5], [3, 0], [6, 9]]))

    # Actual results
    cooc_count.save(path)
    actual = mangoes.CountBasedRepresentation.load(path)

    # Test
    assert os.path.isfile(path)
    assert expected.words.words == actual.words.words
    assert expected.contexts_words.words == actual.contexts_words.words
    try:
        assert np.allclose(expected.matrix, actual.matrix)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected.matrix, actual.matrix)
    assert "x" == actual.params["a"]


@pytest.mark.integration
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_save_load_folder_embeddings(save_temp_dir, matrix_type):
    path = os.path.join(str(save_temp_dir), "test_save")

    words = mangoes.Vocabulary(["a", "b", "c"])
    matrix = matrix_type([[0, 5], [3, 0], [6, 9]])
    embeddings = mangoes.Embeddings(words, matrix)

    # Expected results
    expected = mangoes.Embeddings(mangoes.Vocabulary(["a", "b", "c"]),
                                  matrix_type([[0, 5], [3, 0], [6, 9]]))

    # Actual results
    embeddings.save(path)
    actual = mangoes.Embeddings.load(path)

    # Test
    assert os.path.isdir(path)
    assert expected.words == actual.words
    try:
        assert np.allclose(expected.matrix, actual.matrix)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected.matrix, actual.matrix)


@pytest.mark.integration
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_save_load_archive_embeddings(save_temp_dir, matrix_type):
    path = os.path.join(str(save_temp_dir), "test_save.zip")

    words = mangoes.Vocabulary(["a", "b", "c"])
    matrix = matrix_type([[0, 5], [3, 0], [6, 9]])
    embeddings = mangoes.Embeddings(words, matrix)

    # Expected results
    expected = mangoes.Embeddings(mangoes.Vocabulary(["a", "b", "c"]),
                                  matrix_type([[0, 5], [3, 0], [6, 9]]))

    # Actual results
    embeddings.save(path)
    actual = mangoes.Embeddings.load(path)

    # Test
    assert os.path.isfile(path)
    assert expected.words == actual.words
    try:
        assert np.allclose(expected.matrix, actual.matrix)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected.matrix, actual.matrix)


@pytest.mark.integration
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_save_load_text_files_embeddings(save_temp_dir, matrix_type):
    path = os.path.join(str(save_temp_dir), "test_save.txt")

    words = mangoes.Vocabulary(["a", "b", "c"])
    matrix = matrix_type([[0, 5], [3, 0], [6, 9]])
    embeddings = mangoes.Embeddings(words, matrix)

    # Expected results
    expected = mangoes.Embeddings(mangoes.Vocabulary(["a", "b", "c"]),
                                  matrix_type([[0, 5], [3, 0], [6, 9]]))

    # Actual results
    embeddings.save_as_text_file(path)
    actual = mangoes.Embeddings.load_from_text_file(path)

    # Test
    assert os.path.isfile(path)
    assert expected.words == actual.words
    try:
        assert np.allclose(expected.matrix, actual.matrix)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected.matrix, actual.matrix)


@pytest.mark.integration
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_load_from_one_pickle_embeddings(save_temp_dir, matrix_type):
    path = os.path.join(str(save_temp_dir), "test_load_from.pickle")

    words = ["a", "b", "c"]
    matrix = matrix_type([[0, 5], [3, 0], [6, 9]])
    with open(path, 'wb') as f:
        pickle.dump(words, f)
        pickle.dump(matrix, f)

    # Expected results
    expected = mangoes.Embeddings(mangoes.Vocabulary(["a", "b", "c"]),
                                  matrix_type([[0, 5], [3, 0], [6, 9]]))

    # Actual results
    actual = mangoes.Embeddings.load_from_pickle_files(path)

    assert expected.words == actual.words
    try:
        assert np.allclose(expected.matrix, actual.matrix)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected.matrix, actual.matrix)


@pytest.mark.integration
@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_load_from_two_pickles_embeddings(save_temp_dir, matrix_type):
    words = ["a", "b", "c"]
    vocabulary_file_path = os.path.join(str(save_temp_dir), "test_save_voc.pickle")
    with open(vocabulary_file_path, 'wb') as f:
        pickle.dump(words, f)

    matrix = matrix_type([[0, 5], [3, 0], [6, 9]])
    matrix_file_path = os.path.join(str(save_temp_dir), "test_save_mat.pickle")
    with open(matrix_file_path, 'wb') as f:
        pickle.dump(matrix, f)

    # Expected results
    expected = mangoes.Embeddings(mangoes.Vocabulary(["a", "b", "c"]),
                                  matrix_type([[0, 5], [3, 0], [6, 9]]))

    # Actual results
    actual = mangoes.Embeddings.load_from_pickle_files(matrix_file_path,
                                                       vocabulary_file_path=vocabulary_file_path)

    assert expected.words == actual.words
    try:
        assert np.allclose(expected.matrix, actual.matrix)
    except TypeError:
        assert mangoes.utils.arrays.csrSparseMatrix.allclose(expected.matrix, actual.matrix)


@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_load_embedding_with_representation_archive(save_temp_dir, matrix_type):
    path = os.path.join(str(save_temp_dir), "test_save.zip")
    e = mangoes.Embeddings(mangoes.Vocabulary(["a", "b", "c"]),
                           matrix_type([[0, 5], [3, 0], [6, 9]]))
    e.save(path)

    r = mangoes.base.Representation.load(path)

    assert isinstance(r, mangoes.Embeddings)


@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_load_embedding_with_representation_folder(save_temp_dir, matrix_type):
    path = os.path.join(str(save_temp_dir), "test_save")
    e = mangoes.Embeddings(mangoes.Vocabulary(["a", "b", "c"]),
                           matrix_type([[0, 5], [3, 0], [6, 9]]))
    e.save(path)

    r = mangoes.base.Representation.load(path)

    assert isinstance(r, mangoes.Embeddings)


@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_load_CBR_with_representation_folder(save_temp_dir, matrix_type):
    path = os.path.join(str(save_temp_dir), "test_save_archive.zip")

    words_vocabulary = mangoes.Vocabulary(["a", "b", "c"])
    contexts_vocabulary = mangoes.Vocabulary(["d", "e"])
    matrix = matrix_type([[0, 5], [3, 0], [6, 9]])
    cooc_count = mangoes.CountBasedRepresentation(words_vocabulary, contexts_vocabulary, matrix)
    cooc_count.save(path)

    r = mangoes.base.Representation.load(path)

    assert isinstance(r, mangoes.CountBasedRepresentation)


@pytest.mark.parametrize("matrix_type", [np.array, sparse.csr_matrix], ids=["dense", "sparse"])
def test_load_CBR_with_representation_archive(save_temp_dir, matrix_type):
    path = os.path.join(str(save_temp_dir), "test_save_archive.zip")

    words_vocabulary = mangoes.Vocabulary(["a", "b", "c"])
    contexts_vocabulary = mangoes.Vocabulary(["d", "e"])
    matrix = matrix_type([[0, 5], [3, 0], [6, 9]])
    cooc_count = mangoes.CountBasedRepresentation(words_vocabulary, contexts_vocabulary, matrix)
    cooc_count.save(path)

    r = mangoes.base.Representation.load(path)

    assert isinstance(r, mangoes.CountBasedRepresentation)


def test_load_from_gensim(save_temp_dir):
    import gensim.downloader

    path = os.path.join(str(save_temp_dir), "test_save_gensim.kv")
    gensim_model = gensim.downloader.load('__testing_word2vec-matrix-synopsis')
    gensim_model.save(path)

    mangoes_emb = mangoes.Embeddings.load_from_gensim(path)

    assert len(gensim_model.wv.vocab) == len(mangoes_emb.words)
    for w in mangoes_emb.words:
        np.testing.assert_array_equal(gensim_model.wv.get_vector(w), mangoes_emb[w])


########################
# Exceptions
def test_exception_wrong_separator():
    with pytest.raises(mangoes.utils.exceptions.NotAllowedValue):
        embeddings = mangoes.Embeddings(mangoes.Vocabulary([]), np.matrix([]))
        embeddings.save_as_text_file("here", sep='.')


def test_exception_unknown_transformation():
    source = mock.Mock()
    source.matrix = None
    source.params = {}
    with pytest.raises(mangoes.utils.exceptions.NotAllowedValue):
        mangoes.create_representation(source, reduction="xxx")
