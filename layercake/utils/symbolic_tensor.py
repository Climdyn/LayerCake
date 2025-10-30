
"""

    Symbolic tensor utility module
    ==============================

    Defines useful functions to deal with |Sympy| symbolic tensors.

"""

import numpy as np
from sympy import tensorproduct, tensorcontraction


def symbolic_tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes of two sympy symbolic arrays

    This is based on |Numpy| :func:`~numpy.tensordot` .

    Parameters
    ----------
    a, b: ~sympy.tensor.array.ImmutableDenseNDimArray or ~sympy.tensor.array.MutableDenseNDimArray or ~sympy.tensor.array.ImmutableSparseNDimArray or ~sympy.tensor.array.MutableSparseNDimArray
        Arrays to take the dot product of.

    axes: int or 2-tuple
        If an integer is provided, sum over the last `axes` axes of `a` and the first `axes` axes
        of `b` in order.
        Else, specify the axes to be summed by a 2-tuple of tuples containing the axes.
        The sizes of the corresponding axes must match.

    Returns
    -------
    ~sympy.tensor.array.ImmutableDenseNDimArray or ~sympy.tensor.array.MutableDenseNDimArray or ~sympy.tensor.array.ImmutableSparseNDimArray or ~sympy.tensor.array.MutableSparseNDimArray
        The tensor dot product of the input.

    """
    nda = len(a.shape)
    if isinstance(axes, int):
        a_com = [nda + i for i in range(-axes, 0)]
        b_com = [nda + i for i in range(axes)]
    else:
        a_com = axes[0]
        b_com = [nda + i for i in axes[1]]
    sum_cols = tuple(a_com) + tuple(b_com)

    prod = tensorproduct(a, b)

    return tensorcontraction(prod, sum_cols)


def remove_dic_zeros(dic):
    """Removes zero values from dictionary

    Parameters
    ----------
    dic: dict
        Dictionary which could include zeroes in values to remove.
    Returns
    -------
    dict
        Dictionary with same keys and values as input, but keys with zero value are removed.
    """

    non_zero_dic = dict()
    for key in dic.keys():
        if dic[key] != 0:
            non_zero_dic[key] = dic[key]

    return non_zero_dic


def get_coords_from_index(dic_index, ndim, shape_len):
    """Get the coordinates of a |Sympy| sparse tensor entry along each axis, given its private dictionary index.

    Notes
    -----
    Assumes that every axis has the same length `ndim`.

    Warnings
    --------
    Assumes that the private dictionary has a certain indexing rationale, which may change over time in |Sympy|

    Parameters
    ----------
    dic_index: int
        Index of the sought entry in the sparse tensor private dictionary.
    ndim: int
        The length of the axes.
    shape_len: int
        Rank of the tensor.

    """
    idx = list()
    svv = dic_index * ndim
    for _ in range(shape_len - 1):
        svv = svv / ndim
        idx.append(int(svv % ndim))
        svv -= idx[-1]
    idx.append(int(svv / ndim))
    return tuple(idx[::-1])


def get_coords_and_values_from_tensor(tensor, output='tuple'):
    """Get the coordinates and values of a |Sympy| sparse tensor, as a coordinates-values list.

    Warnings
    --------
    This a function implemented to compensate for the lack of such feature in |Sympy|, and which might need to be
    reimplemented or replaced in the future.

    Parameters
    ----------
    tensor: ~sympy.tensor.array.ImmutableSparseNDimArray or ~sympy.tensor.array.MutableSparseNDimArray
        The tensor from which to return the coordinates and values list.
    output: str
        The kind of output. Can be:

        * `numpy`: return the list as a |Numpy| array.
        * `list`: return the list as nested Python lists.
        * `tuple`: return the list as a list of tuples.

        Default to `tuple`.
    Returns
    -------
    list(list) or list(tuple) or ~numpy.ndarray
        The coordinates-values list.

    """
    ndim = tensor.shape[0]
    shape_len = len(tensor.shape)
    if output == 'numpy':
        n_entries = len(tensor._args[0])
        coo_list = np.zeros((n_entries, shape_len+1), dtype=object)
        i_entries = 0
    else:
        coo_list = list()
    for n, val in tensor._args[0].items():
        coords = get_coords_from_index(n, ndim, shape_len)
        if output == 'tuple':
            coo_list.append((*coords, val))
        elif output == 'list':
            coo_list.append([*coords, val])
        else:
            coo_list[i_entries, :shape_len] = coords
            coo_list[i_entries, shape_len] = val
            i_entries += 1
    return coo_list


def compute_jacobian_permutations(shape):
    """Return the axes permutations needed to compute the Jacobian tensor associated to the symbolic models tendencies' tensor.

    Parameters
    ----------
    shape: tuple
        The shape of the tendencies' tensor.

    Returns
    -------
    list(list(int))
        The list of permutations of the axes needed to compute the models Jacobian matrix.
    """
    n_perm = len(shape) - 2
    permutations = list()
    for i in range(1, n_perm+1):
        perm = [0, i+1,]
        perm += [j for j in range(2, i+1)]
        perm.append(1)
        perm += [j for j in range(i+2, n_perm+2)]
        permutations.append(perm)

    return permutations
