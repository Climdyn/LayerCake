import numpy as np
from sympy import tensorproduct, tensorcontraction


def symbolic_tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes of two sympy symbolic arrays

    This is based on `Numpy`_ :meth:`~numpy.tensordot` .

    .. _Numpy: https://numpy.org/

    Parameters
    ----------
    a, b: ~sympy.tensor.array.DenseNDimArray or ~sympy.tensor.array.SparseNDimArray
        Arrays to take the dot product of.

    axes: int or 2-tuple
        If an integer is provided, sum over the last `axes` axes of `a` and the first `axes` axes
        of `b` in order.
        Else, specify the axes to be summed by a 2-tuple of tuples containing the axes.
        The sizes of the corresponding axes must match.

    Returns
    -------
    output: Sympy tensor
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
        dictionary which could include 0 in values
    Returns
    -------
    dict
        dictionary with same keys and values as input, but keys with value of 0 are removed
    """

    non_zero_dic = dict()
    for key in dic.keys():
        if dic[key] != 0:
            non_zero_dic[key] = dic[key]

    return non_zero_dic


def _get_coords_from_index(dic_index, ndim, shape_len):
    idx = list()
    svv = dic_index * ndim
    for _ in range(shape_len - 1):
        svv = svv / ndim
        idx.append(int(svv % ndim))
        svv -= idx[-1]
    idx.append(int(svv / ndim))
    return tuple(idx[::-1])


def get_coords_and_values_from_tensor(tensor, output='tuple'):
    ndim = tensor.shape[0]
    shape_len = len(tensor.shape)
    if output == 'numpy':
        n_entries = len(tensor._args[0])
        coo_list = np.zeros((n_entries, shape_len+1), dtype=object)
        i_entries = 0
    else:
        coo_list = list()
    for n, val in tensor._args[0].items():
        coords = _get_coords_from_index(n, ndim, shape_len)
        if output == 'tuple':
            coo_list.append((*coords, val))
        elif output == 'list':
            coo_list.append([*coords, val])
        else:
            coo_list[i_entries, :shape_len] = coords
            coo_list[i_entries, shape_len] = val
            i_entries += 1
    return coo_list


