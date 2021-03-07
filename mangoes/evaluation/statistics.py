# -*- coding: utf-8 -*-
"""Functions to evaluate some statistical properties of representations

"""

# ########################################
# Partition function
import numpy as np

import mangoes.utils.arrays


def _default_partition_function(embeddings, vector):
    """Default partition function

    Partition function used as default in function concentration_of_partition_function

    .. math::
        Z_c = \sum{\exp{c.w}}

    """
    return np.sum(np.exp(embeddings.matrix.dot(vector)))


def _generate_random_vectors(nb_vectors, dimension, mu):
    """Generate a list of uniformly random vectors
    """
    random_discourse_vectors = np.zeros(shape=(nb_vectors, dimension))
    for i in range(dimension):
        random_discourse_vectors[:, i] = np.random.normal(size=nb_vectors)
    return [(c / np.linalg.norm(c)) * (np.sqrt(dimension) / 5 / mu) for c in random_discourse_vectors]


def isotropy_from_partition_function(embeddings, discourse_vectors=1000,
                                     partition_function=_default_partition_function, epsilon=0.1):
    """Evaluate the isotropy of the representation computing the values of a partition function

    Compute the values of a repartition function (default: :math:`Z_c = \sum{\exp{cw}}`) for each :math:`c` in a set of
    uniformly random chosen vectors

    References
    ----------
    .. [1] Arora, S., Li, Y., Liang, Y., Ma, T., & Risteski, A. (2015). Rand-walk: A latent variable model approach to
           word embeddings.

    Parameters
    ----------
    embeddings: instance of Embeddings
        The :class:`mangoes.Embeddings` instance to evaluate
    discourse_vectors: int or list, optional
        a list of vectors or the number of vectors to pick uniformly on the sphere of norm 4/µ where µ is the average
        norm of the word vectors. Default : 1000.
    partition_function: callable
        the partition function to evaluate. Default : :func:`._default_partition_function`
    epsilon: float, optional
        determines the interval used to evaluate the concentration : [(1-ε)*mean_value, (1+ε)*mean_value]
        default: 0.1

    Returns
    -------
    tuple
        (concentration i.e. proportion of the values around the mean value, mean_value, values)

    """
    # TODO : only for dense matrices
    try:
        nb_random_vectors = len(discourse_vectors)
    except TypeError:
        nb_random_vectors = discourse_vectors
        discourse_vectors = _generate_random_vectors(nb_random_vectors,
                                                     embeddings.shape[1],
                                                     np.mean(np.linalg.norm(embeddings.matrix, axis=1)))

    partition_values = [partition_function(embeddings, c) for c in discourse_vectors]
    mean_value = np.mean(partition_values)
    nb_around_mean = (((1 - epsilon) * mean_value < partition_values)
                      & (partition_values < (1 + epsilon) * mean_value)).sum()
    return nb_around_mean / nb_random_vectors, mean_value, partition_values


# ########################################
# Distances between words

def _angles(vector, matrix, normalize=True):
    """Compute the angles between a vector and each line of the matrix

    Parameters
    ----------
    vector:
        a vector of dimension d
    matrix:
        a matrix n x d
    normalize: boolean, optional
        set to False if the matrix and the vectors are already normalized (default: True)

    Returns
    -------
    list
        list of angles size n
    """
    if normalize:
        matrix = matrix.normalize()
        vector = vector.normalize()

    cosines = matrix.dot(vector.T)
    try:
        cosines = cosines.todense()
    except AttributeError:
        pass

    # fix rounding errors
    cosines[[cosines > 1]] = 1
    cosines[[cosines < -1]] = -1

    return np.arccos(cosines)


def distances_one_word_histogram(embeddings, word, bins, distance=_angles, normalize=True):
    """Compute the values of an histogram of the distances between a word and all the other words of the Embeddings

    Parameters
    ----------

    embeddings: instance of Embeddings
        The :class:`mangoes.Embeddings` instance to evaluate
    word: str
        a word from the vocabulary of the Embedding
    bins: list
        bin edges, including the rightmost edge.
    distance: callable, optional
        the function to use to compute distances between words. Default: :func:`._angles`
    normalize: boolean, optional
        set to False if the matrix and the vectors are already normalized (default: True)

    Returns
    -------
    array
        an array of size (len(bins) - 1) with the values of the histogram
    """
    word_index = embeddings.words.word_index[word]
    matrix = mangoes.utils.arrays.Matrix.factory(
        np.concatenate((embeddings.matrix[:word_index], embeddings.matrix[word_index + 1:])))
    return np.histogram(distance(embeddings.matrix[word_index], matrix, normalize=normalize), bins=bins)[0]


def distances_histogram(embeddings, bins, distance=_angles, normalize=True):
    """Compute the values of an histogram of the distances between all the words of the Embeddings.

    Parameters
    ----------
    embeddings: instance of Embeddings
        The :class:`mangoes.Embeddings` instance to evaluate
    bins: list
        bin edges, including the rightmost edge.
    distance: callable
        the function to use to compute distances between words
    normalize: boolean
        set to False if the matrix and the vectors are already normalized (default: True)

    Returns
    -------
    array
        an array of size (len(bins) - 1) with the values of the histogram
    """
    if normalize:
        matrix = embeddings.matrix.normalize()
    else:
        matrix = embeddings.matrix

    hist = np.array([0] * (len(bins) - 1))
    for i in range(len(embeddings.words) - 1):
        vector = matrix[i]
        hist += np.histogram(distance(vector, matrix[i + 1:], normalize=False), bins=bins)[0]

    return hist
