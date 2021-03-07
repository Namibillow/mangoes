# -*- coding: utf-8 -*-
"""Weighting transformations to apply to the co-occurrence count matrix.

This module provides transformations that can be used in the `transformations` parameter of
the :func:`mangoes.create_representation()` function to create an Embeddings from a CountBasedRepresentation.

Examples
---------

    >>> import mangoes.base
    >>> ppmi = mangoes.weighting.PPMI(alpha=1)
    >>> embeddings = mangoes.base.create_representation(cc, transformations=ppmi)


See Also
--------
:func:`mangoes.create_representation`
:class:`mangoes.Transformation`
"""

import numpy as np

import mangoes.utils.arrays
import mangoes.utils.exceptions
from mangoes.base import Transformation


class JointProbabilities(Transformation):
    """Defines a transformation that replaces counts with joint probabilities computed from these counts"""

    def __call__(self, matrix):
        """From matrix = [#(wi,cj)], compute the matrix [P(wi,cj)].

        Parameters
        ----------
        matrix: matrix-like object
            values must be non-negative integers

        Returns
        -------
        matrix-like object
        """
        mmatrix = mangoes.utils.arrays.Matrix.factory(matrix)
        return mmatrix / mmatrix.sum(axis=None)


class ConditionalProbabilities(Transformation):
    """Defines a transformation that replaces counts with joint probabilities computed from these counts"""

    def __call__(self, matrix):
        """From matrix = [#(wi,cj)], compute the matrix [P(cj|wi)].

        Assumes that 'matrix' contains only positive values, so that row sum normalization = row l1 normalization.

        Parameters
        ----------
        matrix: matrix-like object
            values must be non-negative integers

        Returns
        -------
        matrix-like object
        """
        return mangoes.utils.arrays.normalize(matrix, norm="l1", axis=1)


class ProbabilitiesRatio(Transformation):
    """Defines a transformation that replaces counts with joint probabilities computed from these counts

    Attributes
    ----------
    alpha: int, optional
        positive number (default=1); "smoothing" parameter for the computation of the context probability
        distributions: P(c_j) = (#c_j)**alpha / sum((#c_k)**alpha)
    """

    def __init__(self, alpha=1):
        super().__init__()
        self._params["alpha"] = alpha

    @property
    def alpha(self):
        return self._params["alpha"]

    def __call__(self, matrix):
        """From matrix = [#(wi,cj)], compute the matrix [P(cj|wi) / P(cj)].

        Parameters
        ----------
        matrix: matrix-like object
            values must be non-negative integers

        Returns
        -------
        matrix-like object
        """
        matrix = mangoes.utils.arrays.Matrix.factory(matrix)

        contexts_counts = matrix.sum(axis=0)
        if self.alpha != 1:
            contexts_counts = np.power(contexts_counts, self.alpha)

        nb_total_contexts = np.sum(contexts_counts)

        if nb_total_contexts > 0:
            contexts_proba = contexts_counts / nb_total_contexts
            contexts_proba[contexts_proba == 0] = 1  # For the upcoming division: this process is allowed
            # because in our case, if the inv_denominator is null then the numerator is theoretically null.
            probabilities = ConditionalProbabilities()
            contexts_words_dep_proba_matrix = probabilities(matrix)
            return contexts_words_dep_proba_matrix / contexts_proba
        else:
            return matrix


class PMI(Transformation):
    """Defines a PMI (Pointwise Mutual Information) transformation

    Attributes
    ----------
    alpha: int, optional
        positive number (default=1); "smoothing" parameter for the computation of the context probability
        distributions: P(c_j) = (#c_j)**alpha / sum((#c_k)**alpha)

    """

    def __init__(self, alpha=1):
        super().__init__()
        self._params["alpha"] = alpha

    @property
    def alpha(self):
        return self._params["alpha"]

    def __call__(self, matrix):
        """From matrix = [#(wi,cj)], compute the 'pmi' matrix [log(P(cj|wi)/P(cj)) or 0 if not defined].

        Parameters
        ----------
        matrix: matrix-like object
            values must be non-negative integers

        Returns
        -------
        matrix-like object
        """
        if self.alpha <= 0:
            raise mangoes.utils.exceptions.NotAllowedValue(msg="'alpha' parameter value must be positive "
                                                               "(value = {})".format(self.alpha))

        contexts_words_proba_ratio_matrix = ProbabilitiesRatio(alpha=self.alpha)(matrix)
        return np.log(contexts_words_proba_ratio_matrix.replace_negative_or_zeros_by(1))


class PPMI(PMI):
    """Defines a PPMI (Positive PMI) transformation.

    Attributes
    ----------
    alpha: int, optional
        positive number (default=1); "smoothing" parameter for the computation of the context probability
        distributions: P(c_j) = (#c_j)**alpha / sum((#c_k)**alpha)

    Returns
    -------
    matrix-like object
    """

    def __init__(self, alpha=1):
        super().__init__(alpha)

    def __call__(self, matrix):
        """From matrix = [#(wi,cj)], compute the 'ppmi' matrix [max(log(P(cj|wi)/P(cj)) or 0 if not defined, 0)].

        Parameters
        ----------
        matrix: matrix-like object
            values must be non-negative integers

        Returns
        -------
        matrix-like object
        """
        return ShiftedPPMI(alpha=self.alpha, shift=1)(matrix)


class ShiftedPPMI(PMI):
    """Defines a Shifted PPMI (Positive PMI) transformation

    From matrix = [#(wi,cj)], compute the matrix [max(log(P(cj|wi)/P(cj))-log(shift) or 0 if not defined, 0)]

    Attributes
    ----------
    alpha: int, optional
        positive number (default=1); "smoothing" parameter for the computation of the context probability
        distributions: P(c_j) = (#c_j)**alpha / sum((#c_k)**alpha)

    shift: int, optional
        number superior or equal to 1 (default=1); The result matrix will be shifted of log(shift)

    Returns
    -------
    matrix-like object
    """

    def __init__(self, alpha=1, shift=1):
        super().__init__(alpha)
        self._params["shift"] = shift

    @property
    def shift(self):
        return self._params["shift"]

    def __call__(self, matrix):
        """From matrix = [#(wi,cj)], compute the matrix [max(log(P(cj|wi)/P(cj))-log(shift) or 0 if not defined, 0)].

        Parameters
        ----------
        matrix: matrix-like object
            values must be non-negative integers

        Returns
        -------
        matrix-like object
        """
        if self.shift < 1:
            msg = "'shift' parameter value must be superior or equal to 1 (value = {})".format(self.shift)
            raise mangoes.utils.exceptions.NotAllowedValue(msg=msg)

        result = PMI(alpha=self.alpha)(matrix) - np.log(self.shift)
        return result.replace_negative_or_zeros_by(0)


class TFIDF(Transformation):
    """Defines a TF-IDF (term frequency-inverse document frequency) transformation"""

    def __call__(self, matrix):
        """From matrix = [#(wi,cj)], computes a tfidf matrix [tf(wi,cj) * idf(wi)]

        * tf(wi,cj) = P(cj|wi) or 0 if not defined
        * idf(wi) = log(#c / #(c,wj)) or 0 if not defined.

        Parameters
        ----------
        matrix: matrix-like object
            values must be non-negative integers

        Returns
        -------
        matrix-like object
        """
        mmatrix = mangoes.utils.arrays.Matrix.factory(matrix)

        tf_matrix = ConditionalProbabilities()(mmatrix)

        nb_contexts = mmatrix.shape[1]
        idf_array = mmatrix.nb_of_non_zeros_values_by_row()
        idf_array[(idf_array != 0)] = np.log(nb_contexts / idf_array[(idf_array != 0)])

        return tf_matrix.multiply_rowwise(idf_array)
