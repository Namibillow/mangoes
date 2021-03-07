# -*- coding: utf-8 -*-
"""Utility classes and functions to handle matrices

This module provides the :class:`Matrix` class that encapsulates all needed methods for both sparse and dense matrices.

"""
import abc
import logging
import os
import pickle
import warnings

import numpy as np
from numpy import dtype
import sklearn.preprocessing
from sklearn.utils import DataConversionWarning
from scipy import sparse

import mangoes.utils.exceptions

logger = logging.getLogger(__name__)
warnings.simplefilter('ignore', DataConversionWarning)


# Because we will normalize and / or center matrices with an 'int' dtype


def sqrt(matrix, inplace=False):
    """Perform element-wise sqrt.

    Parameters
    ----------
    matrix: a matrix-like object
    inplace: boolean, optional
        whether or not to perform the operation inplace (default=False)

    Returns
    -------
    a matrix-like object
        has the same type as the one input
    """
    return matrix.sqrt(inplace=inplace)


def normalize(matrix, norm="l2", axis=1, inplace=False):
    """Normalize the matrix-like input.

    Parameters
    ----------
    matrix: a matrix-like object
    norm: {'l2' (default), 'l1' or 'max'}, optional
    axis: {1 (default) or 0}
        specify along which axis to compute the average values that will be used to carry out the normalization.
        Use axis=1 to normalize the rows, and axis=0 to normalize the columns.
    inplace: boolean
        whether or not to perform inplace normalization (default=False). Note that if the sparse matrix's format is
        not compatible with the axis along which the normalization is asked (csr for axis=1, csc for axis=0), another
        one will be created, and 'inplace' will become moot. Also, will not apply if the input's dtype is 'int' instead
        of a type compatible with division, such as 'float'.

    Returns
    -------
    a matrix-like object

    Warnings
    --------
    the inplace=True parameter isn't implemented yet

    """
    return matrix.normalize(norm=norm, axis=axis, inplace=inplace)


def center(matrix, axis=1, inplace=False):
    """Center the matrix-like input wrt row average, column average, or total average.

    Parameters
    ----------
    matrix: a matrix-like object
    axis: {1 (default), 0, None}
        the axis along which to compute the average vector (ex: if axis=0, the columns are centered, if axis=1,
        then the rows are). If None, the average is computed using all the values of the matrix, and is then subtracted
        from every value.
    inplace: boolean
        whether or not to perform inplace normalization (default=False). Note that if the sparse matrix's format is not
        compatible with the axis along which the normalization is asked (csr for rows=True, csc for Rows=False), another
        one will be created, and 'inplace' will become moot. Also, will not apply if the input's dtype is 'int' instead
        of a type compatible with division, such as 'float'.

    Returns
    -------
    a matrix-like object

    Warnings
    --------

    * the inplace=True parameter isn't implemented yet
    * If the input is a sparse matrix, then a dense numpy array will be created (and the 'inplace' option becomes
      moot), possibly causing a memory exhaustion error.
    """
    return matrix.center(axis=axis, inplace=inplace)


class Matrix:
    """Abstract class used to store generated vectors in a matrix

    """

    @abc.abstractmethod
    def multiply_rowwise(self, array):
        """Multiply the values of the matrix by the value of the array whose index corresponds to the row index

        Examples
        --------
        >>> M = mangoes.utils.arrays.Matrix.factory(np.asarray([[1, 1, 1], [2, 2, 2]]))
        >>> v = [[3], [4]]
        >>> M.multiply_rowwise(v)
        NumpyMatrix([[3, 3, 3], [8, 8, 8]])

        Parameters
        ----------
        array: array
            a 1d array with length = nb of matrix rows

        Returns
        -------
        matrix
            new matrix with same type as self
        """
        pass

    @abc.abstractmethod
    def normalize(self, norm="l2", axis=1, inplace=False):
        """Normalize the matrix.

        Parameters
        ----------
        norm: {'l2' (default), 'l1' or 'max'}
        axis: {1 (default) or 0}
            specify along which axis to compute the average values that will be used to carry out the normalization.
            Use axis=1 to normalize the rows, and axis=0 to normalize the columns.
        inplace: boolean
            whether or not to perform inplace normalization (default=False). Note that if the sparse matrix's format is
            not compatible with the axis along which the normalization is asked (csr for axis=1, csc for axis=0),
            another one will be created, and 'inplace' will become moot. Also, will not apply if the input's dtype
            is 'int' instead of a type compatible with division, such as 'float'.

        Returns
        -------
        a matrix-like object

        Warnings
        --------
        the inplace=True parameter isn't implemented yet

        """
        pass

    @abc.abstractmethod
    def center(self, axis=1, inplace=False):
        """Center the matrix to row average, column average, or total average.

        Parameters
        ----------
        axis: {1 (default), None, 0}
            the axis along which to compute the average vector (ex: if axis=0, the columns  are centered, if axis=1,
            then the rows are). If None, the average is computed using all the values of the matrix, and is then
            subtracted from every value.
        inplace: boolean
            whether or not to perform inplace normalization (default=False). Note that if the sparse matrix's format is
            not compatible with the axis along which the normalization is asked (csr for rows=True, csc for Rows=False),
            another one will be created, and 'inplace' will become moot. Also, will not apply if the input's dtype
            is 'int' instead of a type compatible with division, such as 'float'.

        Returns
        -------
        a matrix-like object

        Warnings
        --------
        * the inplace=True parameter isn't implemented yet
        * If the input is a sparse matrix, then a dense numpy array will be created (and the 'inplace' option becomes
        moot), possibly causing a memory exhaustion error.
        """
        pass

    @abc.abstractmethod
    def sqrt(self, inplace=False):
        """Performs element-wise sqrt.

        Parameters
        ----------
        inplace: boolean
            whether or not to perform the operation inplace (default=False)

        Returns
        -------
        a matrix-like object
        """
        pass

    @abc.abstractmethod
    def combine(self, other, new_shape, row_indices_map=None, col_indices_map=None):
        """Merge another matrix with this one, creating a new matrix

        The shape of the merged matrix is `new_shape`
        The rows and columns of the other matrix are mapped to the columns of the merged matrix using `xxx_indices_map`
        and the resulting matrix is added to the current one, extended to fit the new shape.

        Examples
        --------
        >>> A = NumpyMatrix(np.array(range(6)).reshape((2,3)))
        >>> print(A)
        [[0 1 2]
         [3 4 5]]
        >>> B = NumpyMatrix(np.array(range(4)).reshape((2,2)))
        >>> print(B)
        [[0 1]
         [2 3]]
        >>> A.combine(B, (2, 4), col_indices_map={0:1, 1:3})
        NumpyMatrix([[ 0.,  1.,  2.,  1.],
                     [ 3.,  6.,  5.,  3.]])

        indices_map = {0:1,     # column 0 of the second matrix go to column 1 of the merged matrix
                       1:3}     # column 1 of the second matrix go to column 3 of the merged matrix

        Parameters
        ----------
        other: matrix-like object
            another matrix to merge with this one
        new_shape: tuple
            the shape of the resulting matrix
        row_indices_map: dict
            a mapping between the indices of the rows of 'other' and the merged matrix
        col_indices_map: dict
            a mapping between the indices of the columns of 'other' and the merged matrix

        Returns
        -------
        merged : matrix-like object

        """
        pass

    @abc.abstractmethod
    def pairwise_distances(self, Y=None, metric="cosine", **kwargs):
        """ Compute a distance matrix from the rows of the matrix.

        This method returns the matrix of the distances between the rows of the matrix.
        Or, if `Y` is given (default is None), then the returned matrix is the pairwise distance between the rows of
        the matrix and the ones from Y.

        This function relies on the `sklearn.metrics.pairwise_distances` module so you can use any distance
        available in it.

        Valid values for metric are:
        - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']. These metrics support sparse matrix inputs.
        - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
          'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
          'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

        See the documentation for sklearn.metrics.pairwise_distances for details on these metrics.

        Parameters
        ----------
        Y: matrix, optional
            An optional second matrix.
        metric : string, or callable
            The metric to use when calculating distance between vectors.
        **kwds : optional keyword parameters
            Any further parameters are passed directly to the distance function.

        Returns
        -------
        array [len(self), len(self)] or [len(self), len(Y)]
            A distance matrix D such that D_{i, j} is the distance between the ith and the jth rows of this matrix,
            if Y is None.
            If Y is not None, then D_{i, j} is the distance between the ith row of this matrix and the jth row of Y.
        """
        pass

    @abc.abstractmethod
    def replace_negative_or_zeros_by(self, value):
        """Replace the negative or zeros value of the matrix by a new value

        Parameters
        ----------
        value: int or float

        Returns
        -------
        a matrix-like object
        """
        pass

    @abc.abstractmethod
    def nb_of_non_zeros_values_by_row(self):
        """Return the numbers of non zeros value for each line of the matrix

        Returns
        -------
        np.array
            np.array with number of non zeros values for each row
        """
        pass

    @abc.abstractmethod
    def all_positive(self):
        """Check if all values of the matrix are positive or zero.

        Returns
        -------
        boolean
        """

    @abc.abstractmethod
    def as_dense(self):
        """Return a dense version of the matrix

        Returns
        -------
        array
        """

    @abc.abstractmethod
    def save(self, path, name="matrix"):
        """Save the matrix in a file.

        The format of the file depends of the type of the matrix. See subclasses for more information.

        Parameters
        ----------
        path: str
            path to the folder where the file will be saved
        name: str
            name of the file (without extension)

        Returns
        -------
        str
            Complete path to the created file

        """
        pass

    @classmethod
    def load(cls, path, name):
        """Load a matrix from a '.npz' archive of '.npy' files.

        Parameters
        ----------
        path: str
            path to the folder where the archive is stored
        name: str
            name of the file (without extension)

        Returns
        -------
        a matrix-like object

        """
        try:
            return csrSparseMatrix.load(path, name)
        except FileNotFoundError:
            return NumpyMatrix.load(path, name)

    @abc.abstractstaticmethod
    def format_vector(vector, sep):
        """Format a line of a csrSparseMatrix as a string

        Parameters
        ----------
        vector:
            line of a csrSparseMatrix
        sep: str
            separator to use

        Returns
        -------
        str
            string representation of the vector
        """
        pass

    @staticmethod
    def factory(matrix):
        """Create a Matrix object from a numpy.ndarray or a scipy.sparse.csr_matrix

        If matrix is already a Matrix, returns it

        Parameters
        ----------
        matrix:
            Matrix, numpy.ndarray or scipy.sparse.csr_matrix

        Returns
        -------
        a matrix-like object
        """
        if isinstance(matrix, Matrix):
            return matrix
        if isinstance(matrix, np.ndarray):
            return NumpyMatrix(matrix)
        if isinstance(matrix, sparse.csr_matrix):
            return csrSparseMatrix(matrix)
        if isinstance(matrix, int) or isinstance(matrix, float):
            return matrix

        raise mangoes.utils.exceptions.UnsupportedType("Can't create a matrix from {}".format(type(matrix)))


class csrSparseMatrix(sparse.csr_matrix, Matrix):
    # use of an invalid name because scipy uses the first 3 letters of the class name
    """Class used to store generated vectors in a matrix with a scipy.sparse.csr_matrix

    """
    EXT = '.npz'

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.is_sparse = True

    def multiply_rowwise(self, array):
        nonzero_rows, _ = self.nonzero()
        result = self._with_data(self.data * array[nonzero_rows], copy=True)
        result.eliminate_zeros()
        return csrSparseMatrix(result)

    def __truediv__(self, other):
        import scipy
        if scipy.isscalar(other):
            result = sparse.csr_matrix(self, dtype=np.float64)
            result.data /= other
            result.eliminate_zeros()
            return csrSparseMatrix(result)
        inv_denominator = sparse.csr_matrix(1.0 / other)
        result = self.multiply(inv_denominator)
        result.eliminate_zeros()
        return csrSparseMatrix(result)

    def __sub__(self, other):
        if np.isscalar(other):
            result = self._with_data(self.data - other, copy=True)
            result.eliminate_zeros()
            return result
        return super(csrSparseMatrix, self).__sub__(other)

    def __getitem__(self, item):
        return Matrix.factory(super(csrSparseMatrix, self).__getitem__(item))

    def normalize(self, norm="l2", axis=1, inplace=False):
        return sklearn.preprocessing.normalize(self, norm=norm, axis=axis, copy=not inplace)

    def center(self, axis=1, inplace=False):
        if inplace:
            raise NotImplementedError  # TODO : inplace

        matrix = self.A
        if axis is None:
            return matrix - np.mean(matrix, axis=None)
        else:
            return Matrix.factory(sklearn.preprocessing.scale(matrix, axis=axis, with_mean=True,
                                                              with_std=False, copy=False))

    def sqrt(self, inplace=False):
        if inplace:
            self.data = np.sqrt(self.data)
            return self
        else:
            return self._with_data(np.sqrt(self.data), copy=not inplace)

    def log(self):
        """Natural logarithm, element-wise.

        Adaptation of numpy.log to csr_matrix
        """
        return self._with_data(np.log(self.data), copy=True)

    def replace_negative_or_zeros_by(self, value):
        data = self.data
        data[data <= 0] = value
        result = self._with_data(data, copy=True)
        result.eliminate_zeros()
        return result

    def nb_of_non_zeros_values_by_row(self):
        return np.array(np.diff(self.indptr), dtype=np.float)

    def all_positive(self):
        return np.all(self.data >= 0)

    def combine(self, other, new_shape, row_indices_map=None, col_indices_map=None):
        if not row_indices_map and not col_indices_map:
            return Matrix.factory(self + other)

        get_row = (lambda r: row_indices_map[r]) if row_indices_map else (lambda i: i)
        get_col = (lambda c: col_indices_map[c]) if col_indices_map else (lambda j: j)

        mrg_matrix = self.tolil()
        mrg_matrix.rows = np.resize(mrg_matrix.rows, new_shape[0])
        mrg_matrix.data = np.resize(mrg_matrix.data, new_shape[0])
        for i in range(mrg_matrix.shape[0], new_shape[0]):
            mrg_matrix.rows[i] = []
            mrg_matrix.data[i] = []
        mrg_matrix._shape = new_shape

        for (i, j), v in other.todok().items():
            mrg_matrix[get_row(i), get_col(j)] += v

        return Matrix.factory(mrg_matrix.tocsr())

    def as_dense(self):
        return self.A

    def save(self, path, name='matrix'):
        """Save the matrix as a '.npz' archive of '.npy' files.

        Parameters
        ----------
        path: str
            path to the folder where the archive will be written
        name: str
            name of the file (without extension)

        Returns
        -------
        str
            Complete path to the created file

        """
        file_path = os.path.join(path, name + '.npz')
        np.savez_compressed(file_path, data=self.data, indices=self.indices,
                            indptr=self.indptr, shape=self.shape)
        return file_path

    @classmethod
    def load(cls, path, name):
        """Load a _matrix from a '.npz' archive of '.npy' files.

        Parameters
        ----------
        path: str
            path to the folder where the archive is stored
        name: str
            name of the file (without extension)

        Returns
        -------
        csrSparseMatrix

        """
        loader = np.load(os.path.join(path, name + cls.EXT))
        return cls((loader['data'], loader['indices'], loader['indptr']),
                   shape=loader['shape'])

    @staticmethod
    def load_from_text_file(file_object, nb_columns, data_type, sep):
        """Load a matrix and a list of words from a text file

        Parameters
        ----------
        file_object: file-like object
        nb_columns: int
            number of columns in the matrix
        data_type
        sep: str
            token used as column separator in the text file

        Returns
        -------
        csrSparseMatrix
        """
        # TODO : move to Embeddings
        words = []
        data_list = []
        indices_list = []
        indptr = [0]

        for line in file_object:
            word, *string_numbers = line.strip().split(sep=sep)
            indices, data = zip(*(s.split(":") for s in string_numbers))

            words.append(word)
            data_list.extend(data)
            indices_list.extend(np.array(indices, dtype=dtype('int64')))
            indptr.append(indptr[-1] + len(indices))

        matrix = sparse.csr_matrix((data_list, indices_list, indptr),
                                   shape=((len(words)), nb_columns), dtype=data_type)
        return matrix, words

    @staticmethod
    def format_vector(vector, sep):
        return sep.join("{:d}:{:s}".format(i, str(vector[0, i])) for i in vector.nonzero()[1])

    @staticmethod
    def allclose(first, second):
        """Returns True if two arrays are element-wise equal within a tolerance.git

        Parameters
        ----------
        first: matrix
        second: matrix

        Returns
        -------
        bool
        """
        if not sparse.issparse(first) or not sparse.issparse(second):
            raise mangoes.utils.exceptions.IncompatibleValue("One of the objects is not a scipy.sparse matrix.")
        if first.shape != second.shape:
            return False
        diff_matrix = first - second
        return np.allclose(np.sum(np.abs(diff_matrix.data), None), 0)

    def pairwise_distances(self, Y=None, metric="cosine", **kwargs):
        if Y is not None:
            Y = _check_array(Y, ensure_2d=True)
        if metric == 'chebyshev':
            return sklearn.metrics.pairwise_distances(self, Y, metric=self.chebyshev)

        return sklearn.metrics.pairwise_distances(self, Y, metric=metric, **kwargs)

    @staticmethod
    def chebyshev(X, Y):
        diff = X - Y
        diff.data = np.abs(diff.data)
        return diff.max(axis=1).A


class NumpyMatrix(np.ndarray, Matrix):
    """Class used to store generated vectors in a matrix with a numpy.ndarray

    """
    EXT = '.npy'

    # subclassing np.ndarray : https://docs.scipy.org/doc/np/user/basics.subclassing.html
    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.is_sparse = False
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.is_sparse = getattr(obj, 'is_sparse', None)

    def multiply_rowwise(self, array):
        return self * array

    def normalize(self, norm="l2", axis=1, inplace=False):
        if inplace:
            raise NotImplementedError  # TODO : inplace
        if len(self.shape) == 1:
            # sklearn's normalize function does not work in that case
            matrix = self._check_np_array(self, copy=not inplace,
                                          dtype=(np.float64, np.float32, np.float16))

            if norm == "l2":
                denominator = np.sqrt(np.sum(np.power(matrix, 2)))
            elif norm == "l1":
                denominator = np.sum(np.abs(matrix))
            elif norm == "max":
                denominator = np.max(np.abs(matrix))
            else:
                raise mangoes.utils.exceptions.NotAllowedValue(value=norm, allowed_values=["l1", "l2", "max"])
            if denominator > 0:
                matrix /= denominator
            return matrix
        else:
            return self.__class__(sklearn.preprocessing.normalize(self, norm=norm, axis=axis, copy=not inplace))

    def center(self, axis=1, inplace=False):
        if inplace:
            raise NotImplementedError  # TODO : inplace

        if axis is None:
            return self - np.mean(self, axis=None)
        else:
            return self.__class__(
                sklearn.preprocessing.scale(self, axis=axis, with_mean=True, with_std=False, copy=False))

    def sqrt(self, inplace=False):
        if not inplace:
            return np.sqrt(self)
        else:
            np.sqrt(self, self)
            return self

    def replace_negative_or_zeros_by(self, value):
        result = self.copy()
        result[(result <= 0)] = value
        return result

    def nb_of_non_zeros_values_by_row(self):
        return np.array([(self != 0).sum(axis=1)], dtype=np.float64).T

    def all_positive(self):
        return np.all(self >= 0)

    def combine(self, other, new_shape, row_indices_map=None, col_indices_map=None):
        if not row_indices_map and not col_indices_map:
            merged_matrix = self + other
        else:
            merged_matrix = np.zeros(new_shape)
            if col_indices_map and not row_indices_map:
                for old_index, new_index in col_indices_map.items():
                    merged_matrix[:, new_index] = other[:, old_index]
                merged_matrix[:, :self.shape[1]] += self
            elif row_indices_map and not col_indices_map:
                for old_index, new_index in row_indices_map.items():
                    merged_matrix[new_index, :] = other[old_index, :]
                merged_matrix[:self.shape[0], :] += self
            else:
                for (i, j), value in np.ndenumerate(other):
                    merged_matrix[row_indices_map[i], col_indices_map[j]] = value
                merged_matrix[:self.shape[0], :self.shape[1]] += self

        return Matrix.factory(merged_matrix)

    def as_dense(self):
        return self

    def save(self, path, name='matrix'):
        file_path = os.path.join(path, name + '.npy')
        np.save(file_path, self)
        return file_path

    @staticmethod
    def format_vector(vector, sep):
        return sep.join(str(v) for v in vector)

    @staticmethod
    def load_from_text_file(file_object, sep):
        """Load a matrix and a list of words from a text file

        Parameters
        ----------
        file_object: file-like object
        sep: str
            token used as a separator in the text file

        Returns
        -------
        tuple
            (matrix, list of words)
        """
        # TODO : move to Embeddings
        words = []
        vectors = []

        for line in file_object:
            word, *string_numbers = line.strip().split(sep=sep)
            words.append(word)
            vectors.append(np.array(string_numbers, dtype=np.float))

        matrix = np.array(vectors, dtype=np.float)
        return matrix, words

    @classmethod
    def load(cls, path, name):
        try:
            return np.asarray(np.load(os.path.join(path, name + cls.EXT)))
        except OSError as error:
            # If we have the following error, then the file is probably the result of
            # np.ndarray.dump, which is just a wrapper around a pickle file
            if str(error) == "Failed to interpret file '{}' as a pickle".format(path):
                with open(path, "rb") as dump_file:
                    matrix = np.asarray(pickle.load(dump_file, encoding="latin1"))
                warning_message = """Problem in 'np.load({:s})': in order to try to load
                                  content, the file will now be interpreted as a latin1 encoded
                                  pickle file.
                                  \n\t\t\t\t\tThis can happen if the file is the result of
                                  'np.ndarray.dump()' in python2. To prevent such
                                  compatibility issues,  prefer using 'np.save()' rather
                                  than 'np.ndarray.dump()' if possible."""
                logging.warning(warning_message, path)
                return matrix

    @staticmethod
    def _check_np_array(array, copy=False, dtype="numeric", order=None):
        """Check whether or not a np array can be correctly divided inplace.

        Comes from sklearn's 'check_array' function, but without the constraint of being limited to 2D
        arrays, which will soon becomes cause for exception raising according to sklearn's code.

        Parameters
        ----------
        array: numpy.array
            np array to check / convert
        copy: bool
            whether a forced copy will be triggered. If copy=False (default), a copy might still be triggered by a
            conversion.
        dtype:  str, type, list of types or None
            data type of result (default="numeric").
            If None, the dtype of the input is preserved. If "numeric", dtype is preserved unless array.dtype is object.
            If dtype is a list of types, conversion on the first type is only performed if the dtype of the input is
            not in the list.
        order:  {None (default), 'F', 'C'}
            whether an array will be forced to be fortran or c-style.

        Returns
        -------
        np.ndarray
            a 'correct' version of the input array
        """
        dtype_numeric = dtype == "numeric"

        dtype_orig = getattr(array, "dtype", None)
        if not hasattr(dtype_orig, 'kind'):
            # not a data type (e.g. a column named dtype in a pandas DataFrame)
            dtype_orig = None

        if dtype_numeric:
            if dtype_orig is not None and dtype_orig.kind == "O":
                # if input is object, convert to float.
                dtype = np.float64
            else:
                dtype = None

        if isinstance(dtype, (list, tuple)):
            if dtype_orig is not None and dtype_orig in dtype:
                # no dtype conversion required
                dtype = None
            else:
                # dtype conversion required. Let's select the first element of the
                # list of accepted types.
                dtype = dtype[0]

        array = NumpyMatrix(np.array(array, dtype=dtype, order=order, copy=copy))
        # make sure we acually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)

        return array

    def pairwise_distances(self, Y=None, metric="cosine", **kwargs):
        if Y is not None:
            Y = _check_array(Y, ensure_2d=True)
        return sklearn.metrics.pairwise_distances(self, Y, metric=metric, **kwargs)


def _check_array(array, ensure_2d=False):
    if not sparse.issparse(array):
        array = np.array(array)
        if array.ndim == 1 and ensure_2d:
            array = array.reshape(1, -1)
    return array
