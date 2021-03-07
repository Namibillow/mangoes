# -*- coding: utf-8 -*-
"""Base classes to define and create word representations.

This module provides the abstract base class `Representation` with two implementations and the main function
`create_representation` to construct one..
"""
import abc
import gzip
import logging
import re

import numpy as np
from numpy import dtype
import pandas as pd

import mangoes.evaluation.analogy
import mangoes.utils
import mangoes.utils.arrays
import mangoes.utils.persist
import mangoes.utils.exceptions
from mangoes.constants import ENCODING

_logger = logging.getLogger(__name__)


class Representation(abc.ABC):
    """Abstract base class for a Representation.

    Parameters
    ----------
    words: mangoes.Vocabulary
        words represented as vectors (rows of the matrix)
    matrix: mangoes.utils.arrays.Matrix
        vectors representing words

    """

    def __init__(self, words, matrix, hyperparameters=None):
        self._params = {}
        self.words = words
        self._params["words"] = self.words.params
        self.matrix = mangoes.utils.arrays.Matrix.factory(matrix)
        if hyperparameters:
            self._params.update(hyperparameters)

    def __getitem__(self, word):
        return self.matrix[self.words.index(word)]

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def params(self):
        """Dict of the parameters used to build this matrix, if available"""
        return self._params

    def pprint(self, display=print):
        """Pretty print the matrix with labels for rows and columns"""
        pd.set_option('display.width', None)
        display(self.to_df())

    @abc.abstractmethod
    def to_df(self):
        """Returns a pandas DataFrame representation of this matrix"""
        pass

    @classmethod
    def load(cls, path):
        """Load a Representation

        According to files found in the path, the Representation is either a CountBasedRepresentation object
        or an Embeddings object.

        Parameters
        -----------
        path: str
            Path to a folder or an archive
        """
        try:
            return mangoes.CountBasedRepresentation.load(path)
        except FileNotFoundError:
            return mangoes.Embeddings.load(path)

    def save(self, path):
        """Save the Representation

        Parameters
        ----------
        path: str
            Path to a folder or an archive. Will be created if doesn't exist.
        """
        mangoes.utils.persist.save(self, self._attributes_to_persist, path, self._params)

    def distance(self, word1, word2, metric="cosine", **kwargs):
        """Returns the distance between two words

        This function relies on the `sklearn.metrics.pairwise_distances` module so you can use any distance
        available in it.

        Parameters
        ----------
        word1: str
            First word
        word2: str
            Second word
        metric: str
            The metric to use
        **kwargs : optional keyword parameters
            Any further parameters are passed directly to the distance function.
        """
        return self.pairwise_distances([word1], [word2], metric, **kwargs)[0, 0]

    def pairwise_distances(self, words, other_words=None, metric="cosine", **kwargs):
        """ Compute the distance matrix from one or two list(s) of words.

        This method takes a list of words, and returns the matrix of the distances between them.
        Or, if `other_words` is given (default is None), then the returned matrix is the pairwise
        distance between the words from `words` and the ones from `other_words`.

        This function relies on the `sklearn.metrics.pairwise_distances` module so you can use any distance
        available in it.

        Valid values for metric are:
        - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'].
          These metrics support sparse matrix inputs.
        - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming',
          'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
          'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

        See the documentation for sklearn.metrics.pairwise_distances for details on these metrics.

        Parameters
        ----------
        words : list of str
            List of words
        other_words: list of str, optional
            An optional second list.
        metric : string, or callable
            The metric to use when calculating distance between words vectors.
        **kwargs : optional keyword parameters
            Any further parameters are passed directly to the distance function.

        Returns
        -------
        array [len(words), len(words)] or [len(words), len(other_words)]
            A distance matrix D such that D_{i, j} is the distance between the ith and jth words of the `words` list,
            if `other_words` is None.
            If `other_words` is not None, then D_{i, j} is the distance between the ith word from `words` and the
            jth word from `other_words`.
        """
        vectors = mangoes.utils.arrays.Matrix.factory(self.matrix[self.words.indices(words), :])
        other_vectors = self.matrix[self.words.indices(other_words), :] if other_words is not None else None
        return vectors.pairwise_distances(other_vectors, metric=metric, **kwargs)

    def word_mover_distance(self, sentence1, sentence2, stopwords=None, metric='euclidean', emd=None):
        return mangoes.utils.metrics.word_mover_distance(self, sentence1, sentence2,
                                                         stopwords=stopwords, metric=metric, emd=emd)

    def get_closest_words(self, word, nb=10, metric="cosine"):
        """Returns the closest words and their distances from the given word

        Attributes
        ----------
        word: str, Token or vector
            a word or a vector representing a word
        nb: int
            the number of neighbors to return
        metric: str or callable
            the metric to use (see pairwise_distances method)

        Returns
        -------
        list
            list of tuples (word, distance) sorted by distance
        """
        if isinstance(word, str) or isinstance(word, tuple):
            # word parameter is a word (not a vector)
            distances = self.matrix.pairwise_distances(self[word], metric=metric).flatten()
            first, last = 1, nb + 1
        else:
            # word parameter is a vector
            distances = self.matrix.pairwise_distances(word, metric=metric).flatten()
            first, last = 0, nb

        sorted_indices = np.argsort(distances)
        return [(self.words[i], distances[i]) for i in sorted_indices[first:last]]

    def analogy(self, question, allowed_answers=1):
        # TODO(nami) Support annotated vocabs (e.g. Token()) as well 
        return mangoes.evaluation.analogy.Evaluator(self).predict(question, allowed_answers)


class Embeddings(Representation):
    """Base class for a Word Embedding.

    Parameters
    ----------
    words: mangoes.Vocabulary
        words represented as vectors (rows of the matrix)
    matrix: mangoes.utils.arrays.Matrix
        vectors representing words

    """
    _attributes_to_persist = [('words', mangoes.Vocabulary), ('matrix', mangoes.utils.arrays.Matrix)]

    def __init__(self, words, matrix, hyperparameters=None):
        super().__init__(words, matrix, hyperparameters)

    def __len__(self):
        return len(self.words)

    def to_df(self):
        """Returns a pandas DataFrame representation of this matrix with labels for rows"""
        return pd.DataFrame(self.matrix, index=self.words, dtype=float)

    @classmethod
    def load(cls, path):
        """Load an Embeddings

        This loader expects to find in path :

        * a file named 'matrix.npy' or 'matrix.npz' for the matrix
        * a text files 'words.txt' with the words represented as vectors in the matrix

        Parameters
        -----------
        path: str
            Path to a folder or an archive
        """
        return mangoes.utils.persist.load(cls, cls._attributes_to_persist, path)

    @classmethod
    def load_from_gensim(cls, path):
        """Load an Embeddings from gensim

        Parameters
        ----------
        path: str


        Returns
        -------

        """
        import gensim
        model = gensim.models.KeyedVectors.load(path)
        vocabulary = mangoes.Vocabulary(sorted(model.wv.vocab, key=lambda w: model.wv.vocab[w].index))
        matrix = mangoes.utils.arrays.Matrix.factory(np.array(model.wv.vectors, dtype=float))
        return mangoes.Embeddings(vocabulary, matrix)

    def save_as_text_file(self, file_path, compress=False, sep="\t"):
        """Save the embedding as a text file, with one word and its corresponding list of embedding
        values per line.

        Parameters
        ----------
        file_path: str
            path to the location where to store the Embeddings instance as a text file
        compress: boolean
            whether or not to compress the output text file (default = False).
            If True, it will be compressed using 'gz', and be named accordingly.
        sep: str
            the string that shall act as the delimiter between words and/or between numbers on a line.
            (default = '\t')

        """
        sep_invalid_values = [",", ".", ":"]
        if sep in sep_invalid_values:
            # TODO : check and explain which value are not accepted and why
            error_message = "Wrong 'sep' input: the following values are not accepted: {}. " \
                            "Default value '\t' is recommended.".format(str(sep_invalid_values))
            raise mangoes.utils.exceptions.NotAllowedValue(error_message)

        # Determine whether or not to write a compressed file
        compresslevel = 0
        _get_open_function = lambda: open(file_path, 'w', encoding=ENCODING)
        if compress or file_path.endswith(".gz"):
            compresslevel = 9
            if not file_path.endswith(".gz"):
                file_path = "{:s}{:s}".format(file_path, ".gz")
            _get_open_function = lambda: gzip.open(file_path, 'wt', compresslevel=compresslevel,
                                                   encoding=ENCODING)

        # Write embedding in file
        with _get_open_function() as f:
            if self.matrix.is_sparse:
                f.write("({:d}, {:d}), {dtype:s}\n".format(*self.matrix.shape,
                                                           dtype=repr(self.matrix.dtype)))
            for i, vector in enumerate(self.matrix):
                word = self.words.words[i]
                string_vector = self.matrix.__class__.format_vector(vector, sep)
                f.write("{:s}{:s}{:s}\n".format(word, sep, string_vector))

    @staticmethod
    def load_from_text_file(file_path, sep="\t"):
        """Load an embedding from a text file, where there is one word and its corresponding list
        of embedding values per line.

        The text file may be in a compressed format, such as '.gz'.

        Parameters
        ----------
        file_path: str
            path to the text file containing the Embeddings' instance's data
        sep: str
            the string that shall act as the delimiter between words and/or between numbers on a line.
            (default = '\t')

        Returns
        -------
        Embeddings
        """
        _get_open_function = lambda: open(file_path, 'r', encoding=ENCODING)
        if file_path.endswith(".gz"):
            _get_open_function = lambda: gzip.open(file_path, 'rt', encoding=ENCODING)

        with _get_open_function() as f:
            line = f.readline()
            match = re.match(r"\((\d+), (\d+)\), (dtype.*)", line)
            if match:
                nb_columns = match.group(2)
                data_type = eval(match.group(3))
                matrix, words_list = mangoes.utils.arrays.csrSparseMatrix.load_from_text_file(f, int(nb_columns),
                                                                                              data_type,
                                                                                              sep)
            else:
                f.seek(0)
                matrix, words_list = mangoes.utils.arrays.NumpyMatrix.load_from_text_file(f, sep)

        return Embeddings(mangoes.Vocabulary(words_list), matrix)

    @classmethod
    def load_from_pickle_files(cls, matrix_file_path, vocabulary_file_path=None):
        """Load an Embeddings instance from a matrix and vocabulary pickle file(s).

        Parameters
        ----------
        matrix_file_path: str
            path to the pickle file where is stored at least the matrix (if vocabulary_file_path is not None)
            and also the vocabulary (if vocabulary_file_path is None).
        vocabulary_file_path: str, optional (default=None)
            path to the pickle file, where the vocabulary is stored, if the matrix and the vocabulary are in
            separate files

        Returns
        -------
        Embeddings
        """
        import pickle
        if vocabulary_file_path is None:
            with open(matrix_file_path, "rb") as f:
                object1 = pickle.load(f)
                object2 = pickle.load(f)
                try:
                    matrix = mangoes.utils.arrays.Matrix.factory(object1)
                    words = object2
                except (TypeError, NotImplementedError):
                    matrix = mangoes.utils.arrays.Matrix.factory(object2)
                    words = object1
        else:
            with open(matrix_file_path, "rb") as f:
                matrix = pickle.load(f)
            matrix = mangoes.utils.arrays.Matrix.factory(matrix)
            with open(vocabulary_file_path, "rb") as f:
                words = pickle.load(f)
        return Embeddings(mangoes.Vocabulary(words), matrix)


class CountBasedRepresentation(Representation):
    """Base class for a cooccurrence count matrix.

    Parameters
    ----------
    words_vocabulary: mangoes.Vocabulary
        words represented as vectors (rows of the matrix)
    contexts_vocabulary: mangoes.Vocabulary
        words used as features (columns of the matrix)
    matrix
        numbers of cooccurrence

    """
    _attributes_to_persist = [('words', mangoes.Vocabulary),
                              ('contexts_words', mangoes.Vocabulary),
                              ('matrix', mangoes.utils.arrays.Matrix)]

    def __init__(self, words_vocabulary, contexts_vocabulary, matrix, hyperparameters=None):
        super().__init__(words_vocabulary, matrix, hyperparameters)
        self.contexts_words = contexts_vocabulary
        self._params["contexts_words"] = self.contexts_words.params

    def __iter__(self):
        return self.words.word_index.__iter__()

    def __eq__(self, other):
        # Equality between vocabularies
        if self.words != other.words:
            return False
        if self.contexts_words != other.contexts_words:
            return False
        # Equality between matrix
        return self.matrix == other.matrix

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_df(self):
        """Returns a pandas DataFrame representation of this matrix with labels for rows and columns"""
        if self.matrix.is_sparse:
            return pd.DataFrame.sparse.from_spmatrix(self.matrix,
                                                     index=self.words, columns=self.contexts_words)
        return pd.DataFrame(self.matrix,
                            index=self.words, columns=self.contexts_words,
                            dtype=int)

    @classmethod
    def load(cls, path):
        """Load a CooccurrenceCount

        This loader expects to find in path :

        * a file named 'matrix.npy' or 'matrix.npz' for the matrix
        * two text files 'words.txt' and 'contexts_words.txt' with the words used in rows and columns
          of the matrix, respectively

        Parameters
        -----------
        path: str
            Path to a folder or an archive
        """
        return mangoes.utils.persist.load(cls, cls._attributes_to_persist, path)


class Transformation:
    """Base callable class to define transformation to be applied to a Matrix

    See Also
    --------
    :func:`mangoes.create_representation`
    :mod:`mangoes.weighting`
    :mod:`mangoes.reduction`
    """

    def __init__(self):
        self._params = {"name": self.__class__.__name__}

    @abc.abstractmethod
    def __call__(self, matrix):
        """Apply the transformation and return the transformed matrix

        Parameters
        ----------
        matrix: mangoes.utils.arrays.Matrix

        Returns
        -------
        mangoes.utils.arrays.Matrix
        """
        pass

    @property
    def params(self):
        return self._params


def create_representation(source, weighting=None, reduction=None):
    """Create an Embeddings object from a CooccurrenceCount

    Apply the function(s) passed in weighting and reduction parameters and returns a mangoes.Representation.

    Examples
    --------
    >>>  embedding = mangoes.create_representation(cooccurrence_matrix,
    >>>                                           reduction=mangoes.reduction.pca(dimensions=50))

    Parameters
    ----------
    source: mangoes.CountBasedRepresentation
    weighting: mangoes.Transformation
        weighting function to apply to the source (see : `mangoes.weighting`)
    reduction: mangoes.Transformation
        reduction to apply to the (weighted) source matrix (see : `mangoes.reduction`)

    Returns
    --------
    Embeddings or CountBasedRepresentation

    See Also
    --------
    :mod:`mangoes.weighting`
    :mod:`mangoes.reduction`
    """
    matrix = source.matrix
    hyperparameters = {"source": source.params}

    if weighting:
        if not callable(weighting):
            msg = "`weighting` should be a callable, not a {}".format(type(weighting))
            raise mangoes.utils.exceptions.NotAllowedValue(msg)

        matrix = weighting(matrix)
        hyperparameters["weighting"] = weighting.params

    if reduction:
        if not callable(reduction):
            msg = "`reduction` should be a callable, not a {}".format(type(reduction))
            raise mangoes.utils.exceptions.NotAllowedValue(msg)

        matrix = reduction(matrix)
        hyperparameters["reduction"] = reduction.params
        return mangoes.base.Embeddings(source.words, matrix, hyperparameters)
    else:
        return mangoes.CountBasedRepresentation(source.words, source.contexts_words, matrix, hyperparameters)
