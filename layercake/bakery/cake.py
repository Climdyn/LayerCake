
from contextlib import redirect_stdout
import numpy as np
import sparse as sp
from numba import njit
from layercake.utils.tensor import sparse_mul, jsparse_mul
from sympy import ImmutableSparseNDimArray, MutableSparseNDimArray

real_eps = np.finfo(np.float64).eps


class Cake(object):

    def __init__(self):

        self.layers = list()

    def add_layer(self, layer):
        layer._cake_order = len(self.layers)
        self.layers.append(layer)
        layer._cake = self
        for equation in layer.equations:
            equation._cake = self
            equation.field._cake = self

    @property
    def ndim(self):
        dim = 0
        for layer in self.layers:
            dim += layer.ndim
        return dim

    @property
    def fields_tensor_extent(self):
        extent = dict()
        n = 1
        for layer in self.layers:
            for field in layer.fields:
                ni = n + field.state.__len__()
                extent[field] = (n, ni)
                n = ni
        return extent

    @property
    def number_of_layers(self):
        return self.layers.__len__()

    def compute_tensor(self, numerical=True, compute_inner_products=False):

        for layer in self.layers:
            layer.compute_tensor(numerical, compute_inner_products)

    @property
    def maximum_rank(self):
        max_rank = 0
        for layer in self.layers:
            max_rank = max(max_rank, layer.maximum_rank)
        return max_rank

    @property
    def _layers_first_index(self):
        idx = [1]
        for layer in self.layers[:-1]:
            idx.append(idx[-1] + layer.ndim)
        return idx

    @property
    def _layers_last_index(self):
        idx = list()
        for i, layer in enumerate(self.layers):
            if i == 0:
                idx.append(layer.ndim)
            else:
                idx.append(idx[-1] + layer.ndim + 1)
        return idx

    @property
    def tensor(self):
        shape = tuple([self.ndim + 1] * self.maximum_rank)
        if isinstance(self.layers[0].tensor, sp.COO):
            numerical = True
        elif isinstance(self.layers[0].tensor, ImmutableSparseNDimArray):
            numerical = False
        else:
            raise ValueError('Unable to determine the tensor status of layer 1.')

        if numerical:
            tensor = sp.zeros(shape, dtype=np.float64, format='dok')
        else:
            tensor = MutableSparseNDimArray(iterable={}, shape=shape)

        for i, layer in enumerate(self.layers):
            if (numerical and not isinstance(layer.tensor, sp.COO) or
                    (not numerical and not isinstance(layer.tensor, ImmutableSparseNDimArray))):
                raise ValueError("Your cake is composed of both symbolic and numerical layers. "
                                 "Can't compute the full tensor.")
            lmax = layer.maximum_rank
            if i < self.number_of_layers - 1:
                slices = ([slice(self._layers_first_index[i], self._layers_first_index[i + 1])]
                          + [slice(0, None) for _ in range(lmax - 1)])
                zeros = [0 for _ in range(lmax, len(layer.tensor.shape))]
            else:
                slices = ([slice(self._layers_first_index[i], None)]
                          + [slice(0, None) for _ in range(lmax - 1)])
                zeros = [0 for _ in range(lmax, len(layer.tensor.shape))]
            args = tuple(slices + zeros)
            if numerical:
                tensor[args] = tensor[args] + layer.tensor.todense()[1:]
            else:
                tensor[args] = tensor[args] + layer.tensor[1:]

        if numerical:
            tensor = tensor.to_coo()
            tensor = self.simplify_tensor(tensor)
        else:
            tensor = ImmutableSparseNDimArray(tensor)

        return tensor

    @property
    def jacobian_tensor(self):
        tensor = self.tensor
        if isinstance(tensor, sp.COO):
            return self._jacobian_from_tensor(tensor)
        else:
            return None

    @staticmethod
    def _jacobian_from_tensor(tensor):
        """Function to compute the Jacobian tensor.

        Parameters
        ----------
        tensor: sparse.COO
            The system tensor.

        Returns
        -------
        sparse.COO
            The Jacobian tensor.
        """

        n_perm = len(tensor.shape) - 2

        jacobian_tensor = tensor.copy()

        for i in range(1, n_perm+1):
            jacobian_tensor += tensor.swapaxes(1, i+1)

        return jacobian_tensor

    def compute_tendencies(self):
        if self.tensor is not None:

            if isinstance(self.tensor, sp.COO):
                coo = self.tensor.coords.T
                val = self.tensor.data

                @njit
                def f(t, x):
                    xx = np.concatenate((np.full((1,), 1.), x))
                    xr = sparse_mul(xx, coo, val)
                    return xr[1:]

                jcoo = self.jacobian_tensor.coords.T
                jval = self.jacobian_tensor.data

                @njit
                def Df(t, x):
                    xx = np.concatenate((np.full((1,), 1.), x))
                    mul_jac = jsparse_mul(xx, jcoo, jval)
                    return mul_jac[1:, 1:]

                return f, Df
            else:
                return None, None
        else:
            return None, None

    @staticmethod
    def simplify_tensor(tensor):
        """Routine that simplifies the component of a tensor :math:`\\mathcal{T}`.
        For each index :math:`i`, it upper-triangularizes the
        tensor :math:`\\mathcal{T}_{i,\\ldots}` for all the subsequent indices.

        Parameters
        ----------
        tensor: sparse.COO
            The tensor to simplify.

        Returns
        -------
        sparse.COO
            The upper-triangularized tensor.
        """
        coords = tensor.coords.copy()
        sorted_indices = np.sort(coords[1:, :], axis=0)
        coords[1:, :] = sorted_indices

        upp_tensor = sp.COO(coords, tensor.data.copy(), shape=tensor.shape, prune=True)

        return upp_tensor

    def print_tensor(self, tensor_name=""):
        """Routine to print the tensor.

        Parameters
        ----------
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `Tensor`.
        """
        if not tensor_name:
            tensor_name = 'Tensor'
        for coo, val in zip(self.tensor.coords.T, self.tensor.data):
            self._string_format(print, tensor_name, coo, val)

    def print_tensor_to_file(self, filename, tensor_name=""):
        """Routine to print the tensor to a file.

        Parameters
        ----------
        filename: str
            The filename where to print the tensor.
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `Tensor`.
        """
        with open(filename, 'w') as f:
            with redirect_stdout(f):
                self.print_tensor(tensor_name)

    def print_jacobian_tensor(self, tensor_name=""):
        """Routine to print the Jacobian tensor.

        Parameters
        ----------
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `TensorJacobian`.
        """
        if not tensor_name:
            tensor_name = 'TensorJacobian'
        for coo, val in zip(self.jacobian_tensor.coords.T, self.jacobian_tensor.data):
            self._string_format(print, tensor_name, coo, val)

    def print_jacobian_tensor_to_file(self, filename, tensor_name=""):
        """Routine to print the Jacobian tensor to a file.

        Parameters
        ----------
        filename: str
            The filename where to print the tensor.
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `TensorJacobian`.
        """
        with open(filename, 'w') as f:
            with redirect_stdout(f):
                self.print_jacobian_tensor(tensor_name)

    @staticmethod
    def _string_format(func, symbol, indices, value):
        if abs(value) >= real_eps:
            s = symbol
            for i in indices:
                s += "["+str(i)+"]"
            s += " = % .5E" % value
            func(s)
