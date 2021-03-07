# -*-coding:utf-8 -*
"""This module provides different ways to derive a phrase vector from the vectors of its parts.

Given two vectors :math:`u` and :math:`v` representing two words *w1* and *w2*, composition methods can be applied to
derive a new vector :math:`p` representing the phrase *'w1 w2'* by combining :math:`u` and :math:`v`.

This module provides classes to learn parameters for various compositional models and apply them to predict new vectors.

These models are :

- additive model [#3]_: :math:`p` is obtained by a (weighted) sum of :math:`u` and :math:`v` :

    .. math::
        \mathbf{p = \\alpha u + \\beta v}

  These weights can be learned with :class:`.AdditiveComposer`

- multiplicative [#3]_: :math:`p` is obtained by component-wise multiplication of :math:`u` and :math:`v` :

    .. math::
        \mathbf{p = u \odot v}

  This method has no parameter to learn but :class:`.MultiplicativeComposer` is provided here to be
  compared with other ones.

- dilation model [#2]_ [#3]_: :math:`p` is obtained by calculating the dot products of :math:`u.u` and :math:`u.v`
  and stretching :math:`v` by a factor :math:`\lambda` in the direction of :math:`u` :
    .. math::
        \mathbf{p = (u.u)v + (\lambda - 1)(u.v)}

  :math:`\lambda` can be learned with :class:`.DilationComposer`

- full additive [#4]_: an extension of the additive model where the two n-dimension input vectors are multiplied by
  two n x n weight matrices :
    .. math::
        \mathbf{p = Au + Bv}
  :math:`A` and :math:`B` can be learned with :class:`.FullAdditiveComposer`

- lexical function [#5]_: *w1* is seen as a function and represented as a matrix :math:`U` and :math:`p` is the product
  of this matrix and :math:`v`
    .. math::
        \mathbf{p = Uv}
  :math:`U` can be learned with :class:`.LexicalComposer`

References
----------
.. [#1] Boleda, G., Baroni, M., & McNally, L. (2013). Intensionality was only alleged: On adjective-noun composition in
        distributional semantics. In Proceedings of the 10th International Conference on Computational Semantics
        (IWCS 2013)–Long Papers (pp. 35-46).
.. [#2] Clark, S., Coecke, B., & Sadrzadeh, M. (2008). A compositional distributional model of meaning. In Proceedings
        of the Second Quantum Interaction Symposium (QI-2008) (pp. 133-140).
.. [#3] Mitchell, J., & Lapata, M. (2010). Composition in distributional models of semantics. Cognitive science,
        34(8), 1388-1429.
.. [#4] Guevara, E. (2010, July). A regression model of adjective-noun compositionality in distributional semantics.
        In Proceedings of the 2010 Workshop on GEometrical Models of Natural Language Semantics (pp. 33-37). Association
        for Computational Linguistics.
.. [#5] Baroni, M., & Zamparelli, R. (2010, October). Nouns are vectors, adjectives are matrices: Representing
        adjective-noun constructions in semantic space. In Proceedings of the 2010 Conference on Empirical Methods in
        Natural Language Processing (pp. 1183-1193). Association for Computational Linguistics.

"""

import abc
import logging

import scipy
import numpy as np
import sklearn.cross_decomposition

import mangoes
import mangoes.utils.exceptions

_logger = logging.getLogger(__name__)


class _Composer:
    """Base class for composers"""
    def __init__(self, representation, init=None):
        if representation.matrix.is_sparse:
            # TODO: implement for sparse matrices
            return NotImplementedError
        self.representation = representation
        self.init = init
        self.params = None

    @abc.abstractmethod
    def compose(self, *args):
        pass

    def predict(self, u, v):
        u = self.representation[u]
        v = self.representation[v]
        return self.compose(u, v, *self.params)


class _ScalarComposer(_Composer):
    """Base class for composers learning scalars, using least square regression"""
    def residual(self, parameters, observed, adjectives, nouns):
        return [scipy.spatial.distance.euclidean(o, self.compose(u, v, *parameters))
                for o, u, v in zip(observed, adjectives, nouns)]

    def _fit(self, bigrams=None):
        if not bigrams:
            bigrams = self.representation.words.get_bigrams()
        bigrams = mangoes.Vocabulary(bigrams)

        x = [self.representation[b[0]] for b in bigrams]
        y = [self.representation[b[1]] for b in bigrams]

        observed_vectors = [self.representation[b] for b in bigrams]

        result = scipy.optimize.least_squares(self.residual, self.init, args=(observed_vectors, x, y))
        self.params = result.x

    def transform(self, bigram):
        return self.compose(*bigram)


class AdditiveComposer(_ScalarComposer):
    """Compose phrase vectors using the additive model

    Given two vectors :math:`u` and :math:`v` representing two words *w1* and *w2*, the Additive Model derives
    a new vector :math:`p` representing the phrase *'w1 w2'* by combining :math:`u` and :math:`v` by a (weighted) sum
    of :math:`u` and :math:`v` :

    .. math::
        \mathbf{p = \\alpha u + \\beta v}

    This class learn these weights from a Representation then apply them to predict new vectors

    Examples
    --------
    >>> import numpy
    >>> import mangoes.composition
    >>> colors = ['white', 'black', 'green', 'red']
    >>> nouns = ['dress', 'rabbit', 'flag']
    >>> adj_nouns = ['white dress', 'black dress', 'red flag', 'green flag', 'white rabbit']
    >>> vocabulary = mangoes.Vocabulary(colors + nouns + adj_nouns)
    >>> matrix = numpy.random.random((12, 5))
    >>> embeddings = mangoes.Embeddings(vocabulary, matrix)
    >>> additive_composer = mangoes.composition.AdditiveComposer(embeddings)
    >>> additive_composer.fit()
    >>> green_rabbit = additive_composer.predict('green', 'rabbit')

    Parameters
    ----------
    representation: mangoes.Representation
        Representation from which the weights will be learned. So, the vocabulary represented should contains bigrams
        where both part are also parts of the vocabulary.

    init: (float, float)
        initial values for alpha and beta. Default = (1, 1)

    Attributes
    ----------
    alpha:
        Learned weight applied to first words
    beta:
        Learned weight applied to second words
    """

    def __init__(self, representation, init=(1,1)):
        super().__init__(representation, init)

    @property
    def alpha(self):
        return self.params[0]

    @property
    def beta(self):
        return self.params[1]

    @staticmethod
    def compose(u, v, a, b):
        return a * u + b * v

    def fit(self, bigrams=None):
        """Fit model to data

        Parameters
        ----------
        bigrams: list
            If bigrams = None (default), parameters will be learned from all the bigrams found in the vocabulary of the
            Representation. But a list of bigrams can be provided here : it has to be a subset of the vocabulary of the
            Representation.
        """
        self._fit(bigrams)


class DilationComposer(_ScalarComposer):
    """Compose phrase vectors using the dilation model

    Given two vectors :math:`u` and :math:`v` representing two words *w1* and *w2*, the dilation model derives
    a new vector :math:`p` representing the phrase *'w1 w2'* by calculating the dot products of :math:`u.u` and
    :math:`u.v` and stretching :math:`v` by a factor :math:`\lambda` in the direction of :math:`u` :

    .. math::
        \mathbf{p = (u.u)v + (\lambda - 1)(u.v)}

    This class learn :math:`\lambda` from a Representation then apply the dilation to predict new vectors

    Examples
    --------
    >>> import numpy
    >>> import mangoes.composition
    >>> colors = ['white', 'black', 'green', 'red']
    >>> nouns = ['dress', 'rabbit', 'flag']
    >>> adj_nouns = ['white dress', 'black dress', 'red flag', 'green flag', 'white rabbit']
    >>> vocabulary = mangoes.Vocabulary(colors + nouns + adj_nouns)
    >>> matrix = numpy.random.random((12, 5))
    >>> embeddings = mangoes.Embeddings(vocabulary, matrix)
    >>> dilation_composer = mangoes.composition.AdditiveComposer(embeddings)
    >>> dilation_composer.fit()
    >>> green_rabbit = dilation_composer.predict('green', 'rabbit')

    Parameters
    ----------
    representation: mangoes.Representation
        Representation from which the dilation factor will be learned. So, the vocabulary represented should contains
        bigrams where both part are also parts of the vocabulary.

    init: float
        initial value for lambda_. Default = 0.5

    Attributes
    ----------
    lambda_ : float
        The dilation factor
    """
    def __init__(self, representation, init=0.5):
        super().__init__(representation, init)

    @property
    def lambda_(self):
        return self.params[0]

    @staticmethod
    def compose(u, v, lambda_):
        return (lambda_ - 1) * u.dot(v.T) * u + u.dot(u.T) * v

    def fit(self, bigrams=None):
        """Fit model to data

        Parameters
        ----------
        bigrams: list
            If bigrams = None (default), lambda_ will be learned from all the bigrams found in the vocabulary of the
            Representation. But a list of bigrams can be provided here : it has to be a subset of the vocabulary of the
            Representation.
        """
        self._fit(bigrams)


class MultiplicativeComposer(_Composer):
    """Compose phrase vectors using the multiplicative model

    Given two vectors :math:`u` and :math:`v` representing two words *w1* and *w2*, the multiplicative model derives
    a new vector :math:`p` representing the phrase *'w1 w2'* by component-wise multiplication of :math:`u`
    and :math:`v` :

    .. math::
        \mathbf{p = u \odot v}

    This class has no parameter to learn but is provided with a fit() function to be easily compared with other models.

    Examples
    --------
    >>> import numpy
    >>> import mangoes.composition
    >>> colors = ['white', 'black', 'green', 'red']
    >>> nouns = ['dress', 'rabbit', 'flag']
    >>> adj_nouns = ['white dress', 'black dress', 'red flag', 'green flag', 'white rabbit']
    >>> vocabulary = mangoes.Vocabulary(colors + nouns + adj_nouns)
    >>> matrix = numpy.random.random((12, 5))
    >>> embeddings = mangoes.Embeddings(vocabulary, matrix)
    >>> multiplicative_composer = mangoes.composition.MultiplicativeComposer(embeddings)
    >>> multiplicative_composer.fit() # does nothing
    >>> green_rabbit = multiplicative_composer.predict('green', 'rabbit')

    Parameters
    ----------
    representation: mangoes.Representation
        Representation used to predict phrase vectors.

    """
    def __init__(self, representation):
        super().__init__(representation)

    @staticmethod
    def compose(u, v):
        return u * v

    def fit(self, bigrams=None):
        pass

    def predict(self, u, v):
        u = self.representation[u]
        v = self.representation[v]
        return self.compose(u, v)


class FullAdditiveComposer(_Composer):
    """Compose phrase vectors using the full additive model

    Given two vectors :math:`u` and :math:`v` representing two words *w1* and *w2*, the Full Additive Model derives
    a new vector :math:`p` representing the phrase *'w1 w2'* by multiplying :math:`u` and :math:`v`
    by two n x n weight matrices :

    .. math::
        \mathbf{p = Au + Bv}
    can be learned with :class:`.FullAdditiveComposer`

    This class learn :math:`A` and :math:`B` from a Representation, using PLSR, then apply them to predict new vectors

    Examples
    --------
    >>> import numpy
    >>> import mangoes.composition
    >>> colors = ['white', 'black', 'green', 'red']
    >>> nouns = ['dress', 'rabbit', 'flag']
    >>> adj_nouns = ['white dress', 'black dress', 'red flag', 'green flag', 'white rabbit']
    >>> vocabulary = mangoes.Vocabulary(colors + nouns + adj_nouns)
    >>> matrix = numpy.random.random((12, 5))
    >>> embeddings = mangoes.Embeddings(vocabulary, matrix)
    >>> composer = mangoes.composition.FullAdditiveComposer(embeddings)
    >>> composer.fit()
    >>> green_rabbit = composer.predict('green', 'rabbit')

    Parameters
    ----------
    representation: mangoes.Representation
        Representation from which the weights will be learned. So, the vocabulary represented should contains bigrams
        where both part are also parts of the vocabulary.

    Attributes
    ----------
    A: array
    B: array
        Learned weight applied to first and second words. The shape of the matrices is nxn where n is the
        dimension of the representation
    """
    def __init__(self, representation):
        super().__init__(representation)

    @staticmethod
    def compose(u, v, A, B):
        return A.dot(u) + B.dot(v)

    def fit(self, bigrams=None):
        """Fit model to data

        Parameters
        ----------
        bigrams: list
            If bigrams = None (default), parameters will be learned from all the bigrams found in the vocabulary of the
            Representation. But a list of bigrams can be provided here : it has to be a subset of the vocabulary of the
            Representation.
        """
        if not bigrams:
            bigrams = self.representation.words.get_bigrams()
        bigrams = mangoes.Vocabulary(bigrams)
        pls = sklearn.cross_decomposition.PLSRegression(n_components=self.representation.shape[1]*2, scale=False)

        U = [self.representation[b[0]] for b in bigrams]
        V = [self.representation[b[1]] for b in bigrams]
        observed_vectors = [self.representation[b] for b in bigrams]

        X = np.hstack((U, V))

        pls.fit(X, observed_vectors)
        self.params = pls

        self.A = pls.coef_[:self.representation.shape[1]].T
        self.B = pls.coef_[self.representation.shape[1]:].T


    def predict(self, u, v):
        u = self.representation[u]
        v = self.representation[v]
        return self.params.predict(np.hstack((u, v)).reshape(1, -1))[0]


class LexicalComposer(_Composer):
    """Compose phrase vectors using the lexical model

    Given a word *w1*, the Full Additive Model sees it as a function, represented as a matrix :math:`U` and applies
    it to another word *w2* represented as a vector :math:`v` to derives a new vector :math:`p` representing the
    phrase *'w1 w2'*

    .. math::
        \mathbf{p = Uv}

    This class learn :math:`U` for word *w1* from a Representation, using PLSR, then apply it to predict new vectors

    Examples
    --------
    >>> import numpy
    >>> import mangoes.composition
    >>> colors = ['green']
    >>> nouns = ['dress', 'hat', 'bean', 'lantern', 'rabbit']
    >>> adj_nouns = ['green dress', 'green bean', 'green hat', 'green lantern']
    >>> vocabulary = mangoes.Vocabulary(colors + nouns + adj_nouns)
    >>> matrix = numpy.random.random((10, 5))
    >>> embeddings = mangoes.Embeddings(vocabulary, matrix)
    >>> green = mangoes.composition.LexicalComposer(embeddings, 'green')
    >>> green.fit()
    >>> green_rabbit = green.predict('rabbit')

    Parameters
    ----------
    representation: mangoes.Representation
        Representation from which the weights will be learned. So, the vocabulary represented should contains bigrams
        where both part are also parts of the vocabulary.

    word: str
        The word to represent as a matrix

    n_components: int
        Number of components to keep in PLSR. If None, use 1/4 of the numbers of bigrams that will be used in fit.

    Attributes
    ----------
    U: array
        Learned matrix to represent `word`. The shape of the matrices is nxn where n is the
        dimension of the representation
    """
    def __init__(self, representation, word, n_components=None):
        super().__init__(representation)
        self.word = word
        self.n_components = n_components

    @staticmethod
    def compose(U, v):
        return U.dot(v)

    def fit(self, bigrams=None):
        """Fit model to data

        Parameters
        ----------
        bigrams: list
            If bigrams = None (default), parameters will be learned from all the bigrams found in the vocabulary of the
            Representation. But a list of bigrams can be provided here : it has to be a subset of the vocabulary of the
            Representation.
        """
        if not bigrams:
            bigrams = self.representation.words.get_bigrams()
        bigrams_with_word = [b for b in bigrams
                             if b[0] == self.word and b[1] in self.representation.words]

        if not self.n_components:
            self.n_components = len(bigrams_with_word) // 4
            if self.n_components == 0:
                self.n_components = min(len(bigrams_with_word), self.representation.shape[1])

        y = [self.representation[b[1]] for b in bigrams_with_word]
        observed_vectors = [self.representation[b] for b in bigrams_with_word]

        pls = sklearn.cross_decomposition.PLSRegression(n_components=self.n_components, scale=False)
        pls.fit(y, observed_vectors)

        self.params = pls

    def predict(self, v):
        v = self.representation[v]
        return self.params.predict(v.reshape(1, -1))[0]

    @property
    def U(self):
        return self.params.coef_.T
