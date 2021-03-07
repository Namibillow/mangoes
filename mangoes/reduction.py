# -*- coding: utf-8 -*-
"""Dimensionality reduction to apply the co-occurrence count matrix.

This module provides reductions that can be used in the `transformations` parameter of
the :func:`mangoes.create_representation` function to create an Embeddings from a CooccurrenceCount.

Examples
---------

import mangoes.base    >>> pca = mangoes.reduction.PCA(dimensions=50)
    >>> embeddings = mangoes.base.create_representation(cc, transformations=pca)


See Also
--------
:func:`mangoes.create_representation`
:class:`mangoes.Transformation`
"""
import logging

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sklearn.decomposition

import mangoes.base
import mangoes.utils.exceptions

_logger = logging.getLogger(__name__)


class PCA(mangoes.base.Transformation):
    """Defines a transformation that apply PCA

    Parameters
    ----------
    dimensions: int
        desired dimensionality of the returned matrix

    """
    def __init__(self, dimensions):
        super().__init__()
        self._params["dimensions"] = dimensions

    @property
    def dimensions(self):
        return self._params["dimensions"]

    def __call__(self, matrix):
        """Performs the PCA on the matrix and returns representations for target words

        Given a count matrix, returns a matrix reduced to given dimension using PCA

        Parameters
        ----------
        matrix: a matrix, sparse or dense

        Returns
        -------
        matrix
        """
        # TODO : check performances - power iteration
        # TODO : or alternative (center first ?, optional)
        transformer = sklearn.decomposition.TruncatedSVD(n_components=self.dimensions)
        return transformer.fit_transform(matrix)


class SVD(mangoes.base.Transformation):
    """Defines a transformation to reduce dimensionality using SVD

    Given a count matrix :math:`M`, the SVD decomposition gives :

    .. math::
        M_d = U_d.\Sigma_d.V_d^\\top

    by keeping the top :math:`d` eigenvalues in :math:`\Sigma` where :math:`d` = dimensions

    **If parameter `add_context_vectors` is False (default) :**

    We get :math:`W_d`, a matrix representing the original target words (rows of the matrix) reduced to
    given dimensions :

    .. math::
        W_d = U_d.\Sigma_d^{weight}

    The function will return :math:`W_d`


    **If parameter `add_context_vectors` is True :**

    If the same vocabulary as been used as rows and columns to construct the matrix, you can also set
    `add_context_vectors` to True to add the "context vectors" :math:`C_d` to :math:`W_d`.

    The `symmetric` parameter define the way to construct :math:`C_d` :

    If symmetric = False :

    .. math::
        C_d = V_d.\Sigma_d^{1-weight}

    So weight = 1 corresponds to the traditional SVD factorization :math:`W_d = U_d.\Sigma_d, C_d = V_d`

    If symmetric = True :

    .. math::
        C_d = V_d.\Sigma_d^{weight}

    So with weight = 0 : :math:`W_d = U_d, C_d = V_d`

    The function will return :math:`W_d + C_d`


    Notes
    -----
    `dimensions` has to be lower than both dimensions of the input matrix

    Warnings
    --------

    * `add_context_vectors` should be used **only if** represented words and words used as contexts are the same
    * weight should be a value between 0 and 1

    Attributes
    ----------
    dimensions: int
        desired dimensionality of the returned matrix. Must be less than both dimensions of the original.
    weight: {1, 0, 0.5}
        a parameter that defines the way to compute the matrix (see above)
    add_context_vectors: boolean
        Use the context vectors in addition to the words vectors (should be used **only if** represented words and
        words used as contexts are the same). Default = False.
    symmetric: boolean
        if True, the true matrices :math:`W_d` and :math:`C_d` will be built symmetrically (see above)

    """
    def __init__(self, dimensions, weight=1, add_context_vectors=False, symmetric=False):
        super().__init__()
        self._params["dimensions"] = dimensions
        self._params["weight"] = weight
        self._params["add_context_vectors"] = add_context_vectors
        self._params["symmetric"] = symmetric

    @property
    def dimensions(self):
        return self._params["dimensions"]

    @property
    def weight(self):
        return self._params["weight"]

    @property
    def add_context_vectors(self):
        return self._params["add_context_vectors"]

    @property
    def symmetric(self):
        return self._params["symmetric"]

    def __call__(self, matrix):
        """Performs the reduction

        Parameters
        ----------
        matrix: a matrix, sparse or dense

        Returns
        -------
        matrix
        """
        words_matrix, contexts_matrix = _svd(matrix, self.dimensions, symmetric=self.symmetric, weight=self.weight)

        if self.add_context_vectors:
            return words_matrix + contexts_matrix
        return words_matrix


def _svd(matrix, dimensions, symmetric=False, weight=1):
    """Performs SVD on the matrix and returns representations for target words and context words

    See mangoes.reduction.svd

    Notes
    ------
    `dimensions` has to be lower than both dimensions

    Parameters
    ----------
    matrix: matrix
        can be sparse or dense
    dimensions:
        desired dimensionality of the returned matrix. Must be less than both dimensions of the original.
    symmetric: boolean
        if True, the both matrices Wd and Cd will be built symmetrically (see above)
    weight: {1, 0, 0.5}
        a parameter that defines the way to compute the matrix (see above)

    Returns
    -------
    tuple
        the two matrices : (words_matrix, context_words_matrix)
    """
    # TODO : check performances - power iteration
    if dimensions > min(matrix.shape):
        msg = "'dimensions' parameter must be between 1 and min(matrix.shape). " \
              "dimensions={}, matrix.shape={}".format(dimensions, matrix.shape)
        raise mangoes.utils.exceptions.IncompatibleValue(msg)

    if weight < 0 or weight > 1:
        _logger.warning("weight value should be between 0 and 1 (weight={})".format(weight))

    matrix = matrix.astype(float)
    Ud, sd, Vdt = scipy.sparse.linalg.svds(matrix, k=dimensions)
    Vd = Vdt.T

    if symmetric and weight == 0:
        # no need to construct Sigma
        return Ud, Vd

    Sd_p = np.zeros((dimensions, dimensions))
    Sd_p[:dimensions, :dimensions] = np.diag(sd**weight)

    if symmetric:
        return Ud.dot(Sd_p), Vd.dot(Sd_p)

    Sd_1_p = np.zeros((dimensions, dimensions))
    Sd_1_p[:dimensions, :dimensions] = np.diag(sd**(1 - weight))
    return Ud.dot(Sd_p), Vd.dot(Sd_1_p)
