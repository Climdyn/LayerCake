
"""

    Cake definition module
    ======================

    This module defines the cake object, i.e. the collection of stacked layers
    (represented by :class:`~layercake.bakery.layers.Layer` objects) of a given fluid/media of the system at hand.
    A cake is thus a representation of a layered system defined by the user.
    It also computes and includes the ordinary differential equations representation of the
    partial differential equations contained in the layers, when projected on a given basis (Galerkin procedure).

"""


from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
import sparse as sp
from numba import njit
from layercake.utils.tensor import sparse_mul, jsparse_mul
from layercake.utils.symbolic_tensor import get_coords_and_values_from_tensor, compute_jacobian_permutations
from layercake.formatters.fortran import FortranJacobianEquationFormatter, FortranEquationFormatter
from layercake.formatters.python import PythonJacobianEquationFormatter, PythonEquationFormatter
from layercake.formatters.julia import JuliaJacobianEquationFormatter, JuliaEquationFormatter
from sympy import ImmutableSparseNDimArray, MutableSparseNDimArray
from sympy import simplify, N
from sympy.tensor.array import permutedims

real_eps = np.finfo(np.float64).eps
small_number = 1.e-10


class Cake(object):
    """Class to gather layers of a given fluid/media of the system at hand.

    Attributes
    ----------
    layers: list(~layers.Layer)
        A list of the layer objects included in the cake.

    """

    def __init__(self):

        self.layers = list()

    def add_layer(self, layer):
        """Add a layer object to the cake.

        Parameters
        ----------
        layer: ~layers.Layer
            Layer object to add to the cake.
        """
        layer._cake_order = len(self.layers)
        self.layers.append(layer)
        layer._cake = self
        for equation in layer.equations:
            equation._cake = self
            equation.field._cake = self

    @property
    def ndim(self):
        """int: Dimension of the full ordinary differential equations system of the cake, resulting from
        the Galerkin expansions of the layers."""
        dim = 0
        for layer in self.layers:
            dim += layer.ndim
        return dim

    @property
    def fields(self):
        """list(~field.Field): Returns the list of dynamical fields of the cake, i.e. the fields whose time
        evolution is provided by the partial differential equations of all the layers."""
        fields_list = list()
        for layer in self.layers:
            fields_list += layer.fields
        return fields_list

    @property
    def number_of_equations(self):
        """int: Total number of equations composing the cake."""
        return len(self.fields)

    @property
    def parameters(self):
        """list(~parameter.Parameter): Returns the list of parameters of the cake, i.e. the explicit parameters
        appearing in the partial differential equations of all the layers."""
        parameters_list = list()
        for layer in self.layers:
            for param in layer.parameters:
                if not self._isin(param, parameters_list):
                    parameters_list.append(param)
        return parameters_list

    @staticmethod
    def _isin(o, it):
        res = False
        for i in it:
            if o is i:
                res = True
                break
        return res

    @property
    def parameters_symbols(self):
        """list(~sympy.core.symbol.Symbol): List of parameter's symbols present in all the layers'
        partial differential equations."""
        return [p.symbol for p in self.parameters]

    @property
    def fields_tensor_extent(self):
        """dict(tuple(int)): A dictionary of 2-tuples giving for each dynamical field of the model its entries range in the
        tensor of tendencies."""
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
        """int: Number of layers in the cake."""
        return self.layers.__len__()

    def compute_tensor(self, numerical=True, compute_inner_products=False, compute_inner_products_kwargs=None,
                       substitutions=None, basis_subs=False, parameters_subs=None):
        """Compute the tensor of the symbolic or numerical representation of the ordinary differential
        equations tendencies of all the layers.
        Arguments are passed to the layer :meth:`~Layer.compute_tensor` method.

        Parameters
        ----------
        numerical: bool, optional
            Whether to compute the numerical or the symbolic tensor.
            Default to `True` (numerical tensor as output).
        compute_inner_products: bool, optional
            Whether the inner products tensors of the layer equations' terms must be computed first.
            Default to `False`. Please note that if the inner products are not computed firsthand, the tensor computation
            will fail.
        compute_inner_products_kwargs: dict, optional
            Arguments to pass to the computation of the inner products.
        substitutions: list(tuple), optional
            List of 2-tuples containing extra symbolic substitutions to be made at the end of the tensor computation.
            Only applies for the symbolic tendencies.
            The 2-tuples contain first a |Sympy|  expression and then the value to substitute.
        basis_subs: bool, optional
            Whether to substitute the parameters appearing in the definition of the basis of functions by
            their numerical value.
            Only applies for the symbolic tendencies.
            Default to `False`.
        parameters_subs: list(~parameter.Parameter), optional
            List of model's parameters to substitute in the symbolic tendencies' tensor.
            Only applies for the symbolic tendencies.

        """

        for layer in self.layers:
            layer.compute_tensor(numerical=numerical,
                                 compute_inner_products=compute_inner_products,
                                 compute_inner_products_kwargs=compute_inner_products_kwargs,
                                 substitutions=substitutions,
                                 basis_subs=basis_subs,
                                 parameters_subs=parameters_subs
                                 )

    @property
    def maximum_rank(self):
        """int: Maximum over the ranks of the equations over all the layers."""
        max_rank = 0
        for layer in self.layers:
            max_rank = max(max_rank, layer.maximum_rank)
        return max_rank

    @property
    def _layers_first_index(self):
        """list(int): A list giving for each dynamical field of the model the first index of its entries range in the
        tensor of tendencies."""
        idx = [1]
        for layer in self.layers[:-1]:
            idx.append(idx[-1] + layer.ndim)
        return idx

    @property
    def _layers_last_index(self):
        """list(int): A list giving for each dynamical field of the model the last index of its entries range in the
        tensor of tendencies."""
        idx = list()
        for i, layer in enumerate(self.layers):
            if i == 0:
                idx.append(layer.ndim)
            else:
                idx.append(idx[-1] + layer.ndim + 1)
        return idx

    @property
    def tensor(self):
        """sparse.COO or ~sympy.tensor.array.sparse_ndim_array.ImmutableSparseNDimArray: Return the tensor
        representing the ordinary differential equations tendencies of the whole cake.
        Can be either a numerical or a symbolic representation, depending on the user's choice for the inner products.
        """
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

        tensor = self.simplify_tensor(tensor, small_number)

        return tensor

    @property
    def jacobian_tensor(self):
        """sparse.COO or ~sympy.tensor.array.sparse_ndim_array.ImmutableSparseNDimArray: Return the tensor
        representing the Jacobian matrix of the ordinary differential equations tendencies of the whole cake.
        Can be either a numerical or a symbolic representation, depending on the user's choice for the inner products.
        """
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
        tensor: ~sympy.tensor.array.ImmutableSparseNDimArray
            The system tensor.

        Returns
        -------
        ~sympy.tensor.array.ImmutableSparseNDimArray
            The Jacobian tensor.
        """

        perms = compute_jacobian_permutations(tensor.shape)

        jacobian_tensor = tensor.copy()

        for perm in perms:
            jacobian_tensor += permutedims(tensor, perm)

        return jacobian_tensor

    def compute_tendencies(self, language='python', lang_translation=None, force_symbolic_output=False):
        """Function handling the tendencies tensor to create a tendencies function for the whole cake.
        Returns the tendencies function :math:`\\boldsymbol{f}` determining the model's ordinary differential
        equations:

        .. math:: \\dot{\\boldsymbol{x}} = \\boldsymbol{f}(\\boldsymbol{x})

        It returns also the linearized tendencies
        :math:`\\boldsymbol{\\mathrm{J}} \\equiv \\boldsymbol{\\mathrm{D}f} = \\frac{\\partial \\boldsymbol{f}}{\\partial \\boldsymbol{x}}`
        (Jacobian matrix).

        Depending on whether the tendencies tensor is symbolic or numerical, it will return either
        `Numbafied <https://numba.pydata.org/>`_ callable for the functions :math:`\\boldsymbol{f}` and :math:`J`,
        or list of strings defining each tendency in a computing language selected by the user.

        Parameters
        ----------
        language: str, optional
            String defining in which computing language the tendencies lists must be returned.
            Currently, it can be `'python'`, `'fortran'` or `'julia`'.
            Default to `'python'`.
        lang_translation: dict(str), optional
            Additional language translation mapping provided by the user, mapping replacements for converting
            Sympy symbolic output strings to the target language.
        force_symbolic_output: bool, optional
            Force the return of symbolic tendencies, even if the tensor is numerical.
            Useful to use the results with another language, or to save it in plain text.
            Default to `False`.

        Returns
        -------
        f: callable or list(str), list(Symbol)
            If the tendencies tensor is numerical, the numba-jitted tendencies function.
            If the tendencies tensor is symbolic, or if `force_symbolic` is `True`, the list of tendencies string in the selected target language,
            along with the list of parameters appearing in them.
        Df: callable or list(str), list(Symbol)
            If the tendencies tensor is numerical, the numba-jitted linearized tendencies function.
            If the tendencies tensor is symbolic, or if `force_symbolic` is `True`, the list of linearized tendencies string in the selected target language,
            along with the list of parameters appearing in them.
        """
        if self.tensor is not None:

            if isinstance(self.tensor, sp.COO):
                if force_symbolic_output:
                    t = ImmutableSparseNDimArray(self.tensor.todense())
                    if language == 'python':
                        formatter = PythonEquationFormatter(lang_translation)
                        jacobian_formatter = PythonJacobianEquationFormatter(lang_translation)
                    elif language == 'fortran':
                        formatter = FortranEquationFormatter(lang_translation)
                        jacobian_formatter = FortranJacobianEquationFormatter(lang_translation)
                    elif language == 'julia':
                        formatter = JuliaEquationFormatter(lang_translation)
                        jacobian_formatter = JuliaJacobianEquationFormatter(lang_translation)
                    elif isinstance(language, (tuple, list)):
                        formatter = language[0]
                        jacobian_formatter = language[1]
                    else:
                        raise ValueError('Unable to determine the formatter.')

                    equations_list = formatter(t)
                    jt = ImmutableSparseNDimArray(self.jacobian_tensor.todense())
                    jacobian_equations_list = jacobian_formatter(jt)

                    return (equations_list, t.free_symbols), (jacobian_equations_list, jt.free_symbols)

                else:
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
                elif language == 'julia':
                    formatter = JuliaEquationFormatter(lang_translation)
                    jacobian_formatter = JuliaJacobianEquationFormatter(lang_translation)
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
    def simplify_tensor(tensor, threshold=None):
        """Routine that simplifies the component of a tensor :math:`\\mathcal{T}`.
        For each index :math:`i`, it upper-triangularizes the
        tensor :math:`\\mathcal{T}_{i,\\ldots}` for all the subsequent indices.

        Parameters
        ----------
        tensor: sparse.COO or ~sympy.tensor.array.sparse_ndim_array.ImmutableSparseNDimArray
            The tensor to simplify.
        threshold: float, optional
            If the absolute value of  a tensor entry is lower than this threshold value,
            then this value is removed from the tensor. No threshold is applied if not set.
            Only applies to numerical tensors.
            Useful to filter small spurious results of numerical integrations.

        Returns
        -------
        sparse.COO or ~sympy.tensor.array.sparse_ndim_array.ImmutableSparseNDimArray
            The upper-triangularized tensor.
        """
        if isinstance(tensor, sp.COO):
            coords_val = tensor.coords.copy()
            sorted_indices = np.sort(coords_val[1:, :], axis=0)
            coords_val[1:, :] = sorted_indices

            data = tensor.data.copy()
            if isinstance(threshold, (float, int)):
                data[abs(data) < threshold] = 0.

            upp_tensor = sp.COO(coords_val, data, shape=tensor.shape, prune=True)

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
                    old_val = tensor_dict[coords]
                    new_val = simplify(old_val + val)
                    if new_val != 0:
                        tensor_dict[coords] = new_val
                    else:
                        del tensor_dict[coords]

            return ImmutableSparseNDimArray(iterable=tensor_dict, shape=tensor.shape)
        else:
            raise ValueError('Unable to determine the kind of tensor to simplify.')

    def print_tensor(self, tensor_name="", to_numerics=False):
        """Routine to print the tensor.

        Parameters
        ----------
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `Tensor`.
        to_numerics: bool
            Try to output a numerical value for the tensor entries, even if the tensor is a
            symbolic one.
        """
        if not tensor_name:
            tensor_name = 'Tensor'
        if isinstance(self.tensor, sp.COO):
            for coo, val in zip(self.tensor.coords.T, self.tensor.data):
                self._string_format(print, tensor_name, coo, val)
        elif isinstance(self.tensor, ImmutableSparseNDimArray):
            coords_val = get_coords_and_values_from_tensor(self.tensor, 'tuple')
            for coo_val in coords_val:
                coo = coo_val[:-1]
                val = coo_val[-1]
                if to_numerics:
                    self._string_format(print, tensor_name, coo, N(val))
                else:
                    self._string_format_symbolic(print, tensor_name, coo, val)
        else:
            raise ValueError('Unrecognized tensor format.')

    def print_tensor_to_file(self, filename, tensor_name="", to_numerics=False):
        """Routine to print the tensor to a file.

        Parameters
        ----------
        filename: str
            The filename where to print the tensor.
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `Tensor`.
        to_numerics: bool
            Try to output a numerical value for the tensor entries, even if the tensor is a
            symbolic one.
        """
        with open(filename, 'w') as f:
            with redirect_stdout(f):
                self.print_tensor(tensor_name, to_numerics)

    def print_jacobian_tensor(self, tensor_name="", to_numerics=False):
        """Routine to print the Jacobian tensor.

        Parameters
        ----------
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `TensorJacobian`.
        to_numerics: bool
            Try to output a numerical value for the tensor entries, even if the tensor is a
            symbolic one.
        """
        if not tensor_name:
            tensor_name = 'TensorJacobian'
        if isinstance(self.jacobian_tensor, sp.COO):
            for coo, val in zip(self.jacobian_tensor.coords.T, self.jacobian_tensor.data):
                self._string_format(print, tensor_name, coo, val)
        elif isinstance(self.jacobian_tensor, ImmutableSparseNDimArray):
            coords_val = get_coords_and_values_from_tensor(self.jacobian_tensor, 'tuple')
            for coo_val in coords_val:
                coo = coo_val[:-1]
                val = coo_val[-1]
                if to_numerics:
                    self._string_format(print, tensor_name, coo, N(val))
                else:
                    self._string_format_symbolic(print, tensor_name, coo, val)
        else:
            raise ValueError('Unrecognized Jacobian tensor format.')

    def print_jacobian_tensor_to_file(self, filename, tensor_name="", to_numerics=False):
        """Routine to print the Jacobian tensor to a file.

        Parameters
        ----------
        filename: str
            The filename where to print the tensor.
        tensor_name: str, optional
            Specify the name to print beside the values of the tensor. Default to `TensorJacobian`.
        to_numerics: bool
            Try to output a numerical value for the tensor entries, even if the tensor is a
            symbolic one.
        """
        with open(filename, 'w') as f:
            with redirect_stdout(f):
                self.print_jacobian_tensor(tensor_name, to_numerics)

    def to_latex(self, enclose_lhs=True, drop_first_lhs_char=True, drop_first_rhs_char=False):
        """Generate the LaTeX strings representing the cake's equations mathematically.

        Parameters
        ----------
        enclose_lhs: bool, optional
            Whether to enclose the left-hand side term of the equations inside parenthesis.
            Default to `True`.
        drop_first_lhs_char: bool, optional
            Whether to drop the first two character of the left-hand side latex string of the equations.
            Useful to drop the sign in front of it.
            Default to `True`.
        drop_first_rhs_char: bool, optional
            Whether to drop the first two character of the right-hand side latex string of the equations.
            Useful to drop the sign in front of it.
            Default to `False`.

        Returns
        -------
        dict(list(str))
            The LaTeX strings representing the cake's equations.
            It is a dictionary with one entry per cake's layer.
        """

        latex_string_dict = dict()

        for i, layer in enumerate(self.layers):
            latex_string_list = layer.to_latex(enclose_lhs=enclose_lhs,
                                               drop_first_lhs_char=drop_first_lhs_char,
                                               drop_first_rhs_char=drop_first_rhs_char
                                               )

            latex_string_dict[i] = latex_string_list

        return latex_string_dict

    def show_latex(self, enclose_lhs=True, drop_first_lhs_char=True, drop_first_rhs_char=False):
        """Show the LaTeX string representing the cake's equations mathematically rendered in a window.

        Parameters
        ----------
        enclose_lhs: bool, optional
            Whether to enclose the left-hand side term of the equations inside parenthesis.
            Default to `True`.
        drop_first_lhs_char: bool, optional
            Whether to drop the first two character of the left-hand side latex string of the equations.
            Useful to drop the sign in front of it.
            Default to `True`.
        drop_first_rhs_char: bool, optional
            Whether to drop the first two character of the right-hand side latex string of the equations.
            Useful to drop the sign in front of it.
            Default to `False`.
        """

        latex_string_dict = self.to_latex(enclose_lhs=enclose_lhs,
                                          drop_first_lhs_char=drop_first_lhs_char,
                                          drop_first_rhs_char=drop_first_rhs_char
                                          )

        plt.figure(figsize=(8, self.number_of_equations))
        plt.axis('off')
        k = 0
        number_of_lines = self.number_of_equations + self.number_of_layers
        for i in latex_string_dict:
            if self.layers[i].name:
                plt.text(-0.1, (number_of_lines - k) / (number_of_lines + 1), f'Layer {i} ({self.layers[i].name}):')
            else:
                plt.text(-0.1, (number_of_lines - k) / (number_of_lines + 1), f'Layer {i}:')
            k += 1
            for s in latex_string_dict[i]:
                plt.text(-0.1, (number_of_lines - k) / (number_of_lines + 1), '$%s$' % s)
                k += 1
        plt.show()

    @staticmethod
    def _string_format(func, symbol, indices, value):
        """String formatting for the numerical tensor printing."""
        if abs(value) >= real_eps:
            s = symbol
            for i in indices:
                s += "["+str(i)+"]"
            s += " = % .5E" % value
            func(s)

    @staticmethod
    def _string_format_symbolic(func, symbol, indices, value):
        """String formatting for the symbolic tensor printing."""
        s = symbol
        for i in indices:
            s += "["+str(i)+"]"
        s += f" = {value}"
        func(s)
