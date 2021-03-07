# -*- coding: utf-8 -*-
"""Utility metrics functions.

"""

import logging

import scipy
import numpy as np
import sklearn.metrics

import mangoes.utils.arrays
import mangoes.utils.exceptions

logger = logging.getLogger(__name__)


def rowwise_cosine_similarity(A, B):
    """Compute cosine_similarity between each corresponding rows of A and B

    Parameters
    ----------
    A: matrix-like object
    B: matrix-like object

    Returns
    -------
    list of float
    """
    if len(A.shape) == 1:
        return sklearn.metrics.pairwise.cosine_similarity(A.reshape((1, -1)), B.reshape((1, -1)))

    # TODO : unify sparse and non sparse
    import scipy.sparse
    if scipy.sparse.issparse(A):
        return np.array(
            [sklearn.metrics.pairwise.cosine_similarity(a.toarray().reshape((1, -1)),
                                                        b.toarray().reshape((1, -1)))[0][0]
             for (a, b) in zip(A, B)])

    return np.array(
        [sklearn.metrics.pairwise.cosine_similarity(a.reshape((1, -1)), b.reshape((1, -1)))[0][0]
         for (a, b) in zip(A, B)])


def pairwise_non_negative_cosine_similarity(first, second, normalize=True):
    """Compute non negative cosine similary between all pairs of vectors in matrices

    Parameters
    ----------
    first: matrix-like object
        :class:`mangoes.utils.arrays.Matrix` with n vectors
    second: matrix-like object
        :class:`mangoes.utils.arrays.Matrix` with k vectors
    normalize: bool
        the matrices have to be normalized : if they both are, set this parameter to False

    Returns
    -------
    matrix-like object
        :class:`mangoes.utils.arrays.Matrix` of shape (n x k)
    """
    if normalize:
        first, second = first.normalize(), second.normalize()

    similarities = first.dot(second.T)  # TODO : check if already normalized
    try:
        return mangoes.utils.arrays.Matrix.factory((similarities + 1) / 2)
    except NotImplementedError:
        # adding a nonzero scalar to a sparse matrix is not yet supported
        return mangoes.utils.arrays.Matrix.factory((similarities.todense() + 1) / 2)


def pairwise_cosine_similarity(first, second, normalize=True):
    """Compute cosine similary between all pairs of vectors in matrices

    Parameters
    ----------
    first: matrix-like object
        a mangoes.utils.arrays.Matrix with n vectors
    second: matrix-like object
        a mangoes.utils.arrays.Matrix with k vectors
    normalize: bool
        the matrices have to be normalized : if they both are, set this parameter to False

    Returns
    -------
    matrix-like object
        mangoes.utils.arrays.Matrix of shape (n x k)
    """
    if normalize:
        first, second = first.normalize(), second.normalize()

    similarities = first.dot(second.T)  # TODO : check if already normalized

    import scipy.sparse  # TODO : try/except
    if scipy.sparse.issparse(similarities):
        similarities = similarities.todense()
    return np.array(similarities)


def _earth_mover_distance(d1, d2, cost):
    """Compute Earth Mover's distance between 2 distributions

    Notes
    -----
    This naive implementation is used as a fallback if pyemd is not installed but can't handle large distributions.

    References
    ----------
    Y. Rubner, C. Tomasi, and L. J. Guibas. "A metric for distributions with applications to image databases."
    In IEEE International Conference on Computer Vision, pages 59-66, January 1998.

    Parameters
    ----------
    d1: list
    d2: list
        Distributions to compare, of sizes n1 and n2
    cost: matrix
        Matrix n1 x n2 with the costs to travel from each point in d1 to each point in d2

    Returns
    -------
    (float, matrix)
        The computed distance and the flow matrix

    """

    def _build_eq_constraint_matrix(n1, n2):
        """Build a matrix to compute equality constraints on rows and columns to be used in scipy.optimize.linprog

        Examples
        ---------
        If n1 = n2 = 2
        The (flatten) flow matrix to build is F = [t0 t1
                                                   t2 t3]
        d1 = [x0 x1]
        d2 = [y0 y1]

        The constraints are :
        - on the rows : t0+t1=x0 and t2+t3=x1
        - on the columns : t0+t2=y0 and t1+t3=y1

        So the matrix is : A = [[1 1 0 0],
                                [0 0 1 1],
                                [1 0 1 0],
                                [0 1 0 1]]
        So A.F = [t0 + t1,
                  t2 + t3,
                  t0 + t2,
                  t1 + t3]
        """

        A = np.zeros((n1 + n2, n1 * n2), dtype=float)
        for i in range(n1):
            # the sum of the values of each row of the flow matrix has to be equal to the corresponding value in d1
            for j in range(n1):
                A[i, i * n1 + j] = 1
        for i in range(n2):
            # the sum of the values of each column of the flow matrix has to be equal to the corresponding value in d2
            for j in range(n2):
                A[i + n2, i + n2 * j] = 1
        return A

    n1, n2 = len(d1), len(d2)

    A_eq = _build_eq_constraint_matrix(n1, n2)
    b_eq = np.array([d1, d2], dtype=float).flatten()

    result = scipy.optimize.linprog(cost.flatten(), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (n1 * n2),
                                    method='simplex')

    if not result.success:
        logger.error("\n" + str(result))
        raise mangoes.utils.exceptions.RuntimeError("Local implementation of earth mover's distance failed. You "
                                                    "should install POT or pyemd.")

    flow_matrix = result.x.reshape((n1, n2))
    return (flow_matrix * cost).sum(), flow_matrix


def word_mover_distance(representation, sentence1, sentence2,
                        stopwords=None, metric="euclidean",
                        return_flow=False, emd=None):
    """Compute the Word Mover's Distance between two phrases

    References
    ----------
    Matt Kusner et al. "From Word Embeddings To Document Distances"

    POT: Python Optimal Transport : http://pot.readthedocs.io
    pyemd : https://github.com/wmayner/pyemd

    Examples
    --------
    >>> representation = mangoes.Embedding(...)
    >>> sentence_obama = 'Obama speaks to the media in Illinois'
    >>> sentence_president = 'The president greets the press in Chicago'
    >>> import nltk.corpus
    >>> stopwords = nltk.corpus.stopwords.words('english')
    >>> distance = mangoes.utils.metrics.word_mover_distance(representation,
    >>>                                                      sentence_obama, sentence_president, stopwords=stopwords)

    Parameters
    ----------
    representation: mangoes.Representation
        Words vectors
    sentence1: str or list of str
    sentence2: str or list of str
        The two sentences, phrases or documents to compare
    stopwords: list of str (optional)
        List of words to ignore
    metric: str
        Metric to use to compute distances between words (see Representation.pairwise_distances())
    return_flow: boolean (optional)
        If True, returns the flow matrix and the corresponding words with the distance
    emd: None (default), "pot", "pyemd" or callable
        Implementation of Earth Mover's distance to use.
        If None, try to import `POT` or `pyemd` and use it. If neither `POT` nor `pyemd` is installed, use the
        implementation in this module.
        You can also use your own implementation (see _earth_mover_distance())

    Returns
    -------
    float or (float, dict)
        Returns the computed word mover distance.
        If return_flow, also returns a dictionary with the outgoing flows from the words in sentence1 to the words
        of sentence2

    """

    def _check_emd(emd):
        if not emd:
            emd = _get_default_emd()

        if callable(emd):
            return emd

        if emd in {"POT", "pot", "ot"}:
            return _emd_with_pot
        elif emd == "pyemd":
            return _emd_with_pyemd
        else:
            return _earth_mover_distance

    def _get_default_emd():
        import importlib
        if importlib.util.find_spec("ot"):
            return 'pot'
        if importlib.util.find_spec("pyemd"):
            return 'pyemd'

        logger.warning(
            "neither pot or pyemd is installed : a local implementation will be used instead but "
            "can't handle large sentences : you should consider installing pot or pyemd")
        return _earth_mover_distance

    def _emd_with_pot(d1, d2, words_distances):
        import ot
        distance, log = ot.emd2(d1, d2, words_distances, return_matrix=True)
        flow_matrix = log['G']
        return distance, flow_matrix

    def _emd_with_pyemd(d1, d2, words_distances):
        import pyemd
        distance, flow_matrix = pyemd.emd_with_flow(d1, d2, words_distances)
        return distance, flow_matrix

    def _prepare_sentence(sentence):
        if isinstance(sentence, str):
            sentence = sentence.split()
        return [w for w in sentence if w in representation.words]

    def _sentence_to_nbow_vector(sentence, words):
        nbow = np.zeros((len(words),))
        for w in sentence:
            try:
                nbow[words.index(w)] += 1
            except KeyError:
                pass  # stopwords
        return nbow / sum(nbow)

    sentence1, sentence2 = _prepare_sentence(sentence1), _prepare_sentence(sentence2)
    words = set(sentence1 + sentence2)
    if stopwords:
        words -= set(stopwords)
    words = mangoes.Vocabulary(list(words))

    if len(sentence1) == 0 or len(sentence2) == 0 or len(words) == 0:
        # TODO : log a warning
        return np.nan

    d1, d2 = _sentence_to_nbow_vector(sentence1, words), _sentence_to_nbow_vector(sentence2, words)

    words_distances = representation.pairwise_distances(words, metric=metric)

    emd = _check_emd(emd)
    distance, flow_matrix = emd(d1, d2, words_distances)

    if not return_flow:
        return distance

    flow = {}
    for w in sentence1:
        if w in words:
            flow[w] = {(words[j], f) for j, f in enumerate(flow_matrix[words.index(w)]) if f}

    return distance, flow
