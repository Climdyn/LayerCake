
import numpy as np
from numba import njit


@njit
def sparse_mul(vec, coo, value):
    """Sparse multiplication of a tensor with a vector, resulting into a vector.

    Warnings
    --------
    It is a Numba-jitted function, so it cannot take a :class:`sparse.COO` sparse tensor directly.
    The tensor coordinates list and values must be provided separately by the user.

    Parameters
    ----------
    vec: ~numpy.ndarray(float)
        The vector to contract the tensor with. Must be of shape (`ndim`,) where `ndim` is the second dimension of the tensor.
    coo: ~numpy.ndarray(int)
        A 2D array of shape (n_elems, rank), a list of n_elems tensor coordinates corresponding to each value provided, and rank the rank of the tensor.
    value: ~numpy.ndarray(float)
        A 1D array of shape (n_elems,), a list of value in the tensor

    Returns
    -------
    ~numpy.ndarray(float)
        The result vector of shape (`ndim`,).
    """
    res = np.zeros_like(vec)
    n_elems = coo.shape[0]
    rank = coo.shape[1]
    for n in range(n_elems):
        prod = value[n]
        for i in range(1, rank):
            prod = prod * vec[coo[n, i]]
        res[coo[n, 0]] += prod
    return res


@njit
def jsparse_mul(vec, coo, value):
    """Sparse multiplication of a tensor with a vector, resulting in a matrix.

    Warnings
    --------
    It is a Numba-jitted function, so it cannot take a :class:`sparse.COO` sparse tensor directly.
    The tensor coordinates list and values must be provided separately by the user.

    Parameters
    ----------
    vec: ~numpy.ndarray(float)
        The vector to contract the tensor with. Must be of shape (`ndim`,) where `ndim` is the second dimension of the tensor.
    coo: ~numpy.ndarray(int)
        A 2D array of shape (n_elems, rank), a list of n_elems tensor coordinates corresponding to each value provided, and rank the rank of the tensor.
    value: ~numpy.ndarray(float)
        A 1D array of shape (n_elems,), a list of value in the tensor.

    Returns
    -------
    ~numpy.ndarray(float)
        The resulting tensor.
    """

    n_elems = coo.shape[0]
    rank = coo.shape[1]
    res = np.zeros((len(vec), len(vec)))

    for n in range(n_elems):
        prod = value[n]
        for i in range(2, rank):
            prod = prod * vec[coo[n, i]]
        res[coo[n, 0], coo[n, 1]] += prod

    return res
