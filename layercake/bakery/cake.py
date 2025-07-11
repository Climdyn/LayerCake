
from contextlib import redirect_stdout
import numpy as np
import sparse as sp
from numba import njit
from layercake.utils.tensor import sparse_mul, jsparse_mul
from layercake.utils.symbolic_tensor import get_coords_and_values_from_tensor, compute_jacobian_permutations
from layercake.formatters.fortran import FortranJacobianEquationFormatter, FortranEquationFormatter
from layercake.formatters.python import PythonJacobianEquationFormatter, PythonEquationFormatter
from sympy import ImmutableSparseNDimArray, MutableSparseNDimArray
from sympy import simplify
from sympy.tensor.array import permutedims

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
        else:
            tensor = ImmutableSparseNDimArray(tensor)

        tensor = self.simplify_tensor(tensor)

        return tensor

    @property
    def jacobian_tensor(self):
        tensor = self.tensor
        if isinstance(tensor, sp.COO):
            return self._jacobian_from_numerical_tensor(tensor)
        elif isinstance(tensor, ImmutableSparseNDimArray):
            return self._jacobian_from_symbolic_tensor(tensor)
        else:
            raise ValueError('Unable to determine the kind of tensor to simplify.')

    @staticmethod
    def _jacobian_from_numerical_tensor(tensor):
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

    @staticmethod
    def _jacobian_from_symbolic_tensor(tensor):
        """Function to compute the Jacobian tensor.

        Parameters
        ----------
        tensor: ~sympy.tensor.array.sparse_ndim_array.ImmutableSparseNDimArray
            The system tensor.

        Returns
        -------
        ~sympy.tensor.array.sparse_ndim_array.ImmutableSparseNDimArray
            The Jacobian tensor.
        """

        perms = compute_jacobian_permutations(tensor.shape)

        jacobian_tensor = tensor.copy()

        for perm in perms:
            jacobian_tensor += permutedims(tensor, perm)

        return jacobian_tensor

    def compute_tendencies(self, language='python', lang_translation=None):
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

            elif isinstance(self.tensor, ImmutableSparseNDimArray):
                if language == 'python':
                    formatter = PythonEquationFormatter(lang_translation)
                    jacobian_formatter = PythonJacobianEquationFormatter(lang_translation)
                elif language == 'fortran':
                    formatter = FortranEquationFormatter(lang_translation)
                    jacobian_formatter = FortranJacobianEquationFormatter(lang_translation)
                elif isinstance(language, (tuple, list)):
                    formatter = language[0]
                    jacobian_formatter = language[1]
                else:
                    raise ValueError('Unable to determine the formatter.')

                t = self.tensor
                equations_list = formatter(t)
                jt = self.jacobian_tensor
                jacobian_equations_list = jacobian_formatter(jt)

                return (equations_list, t.free_symbols), (jacobian_equations_list, jt.free_symbols)

            else:
                raise ValueError('Something went very wrong. Unable to determine the kind of the tensor.')
        else:
            raise ValueError("You must first compute the tensor of your cake before computing the tendencies."
                             "Run the 'compute_tensor' method first.")

    @staticmethod
    def simplify_tensor(tensor):
        """Routine that simplifies the component of a tensor :math:`\\mathcal{T}`.
        For each index :math:`i`, it upper-triangularizes the
        tensor :math:`\\mathcal{T}_{i,\\ldots}` for all the subsequent indices.

        Parameters
        ----------
        tensor: sparse.COO or ~sympy.tensor.array.sparse_ndim_array.ImmutableSparseNDimArray
            The tensor to simplify.

        Returns
        -------
        sparse.COO or ~sympy.tensor.array.sparse_ndim_array.ImmutableSparseNDimArray
            The upper-triangularized tensor.
        """
        if isinstance(tensor, sp.COO):
            coords_val = tensor.coords.copy()
            sorted_indices = np.sort(coords_val[1:, :], axis=0)
            coords_val[1:, :] = sorted_indices

            upp_tensor = sp.COO(coords_val, tensor.data.copy(), shape=tensor.shape, prune=True)

            return upp_tensor
        elif isinstance(tensor, ImmutableSparseNDimArray):
            coords_val = get_coords_and_values_from_tensor(tensor, 'numpy')
            sorted_indices = np.sort(coords_val[:, 1:-1], axis=1)
            coords_val[:, 1:-1] = sorted_indices
            tensor_dict = dict()
            for cv in coords_val:
                coords = tuple(cv[:-1])
                val = cv[-1]
                if coords not in tensor_dict:
                    tensor_dict[coords] = val
                else:
                    old_val = tensor[coords]
                    new_val = simplify(old_val + val)
                    if new_val != 0:
                        tensor_dict[coords] = new_val

            return ImmutableSparseNDimArray(iterable=tensor_dict, shape=tensor.shape)
        else:
            raise ValueError('Unable to determine the kind of tensor to simplify.')

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
