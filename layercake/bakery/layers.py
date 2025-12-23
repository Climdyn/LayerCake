
"""

    Layer definition module
    =======================

    This module defines the layer object, i.e. the collection of partial differential equations
    (represented by :class:`~layercake.arithmetic.equation.Equation` objects) which governs the time evolution of
    a given fluid/media layer of the system at hand.
    It also computes and includes the ordinary differential equations representation of the
    partial differential equations, when projected on a given basis (Galerkin procedure).

"""


import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import sparse as sp
from sympy import MutableSparseNDimArray, MutableSparseMatrix, ImmutableMatrix, ImmutableSparseNDimArray
from sympy import zeros as sympy_zeros
from sympy.matrices.exceptions import NonInvertibleMatrixError
from layercake.arithmetic.terms.constant import ConstantTerm
from layercake.arithmetic.terms.operations import ProductOfTerms
from layercake.variables.field import ParameterField, FunctionField
from layercake.utils.symbolic_tensor import symbolic_tensordot


class Layer(object):
    """Class to gather partial differential equations modelling
    a given fluid/media layer of the system at hand.

    Parameters
    ----------
    name: str, optional
        Optional name for the layer.

    Attributes
    ----------
    equations: list(~equation.Equation)
        A list of the equation objects included in the layer.
    tensor: sparse.COO or ~sympy.tensor.array.ImmutableSparseNDimArray
        The tensor representing the ordinary differential equations tendencies.
        Can be either a numerical or a symbolic representation, depending on the user's choice.
    name: str
        Optional name for the layer.
    """

    def __init__(self, name=''):

        self.equations = list()
        self.tensor = None
        self.name = name
        self._cake = None
        self._cake_order = 0

    @property
    def _cake_first_index(self):
        if self._cake is not None:
            return self._cake._layers_first_index[self._cake_order]
        else:
            return None

    @property
    def _cake_last_index(self):
        if self._cake is not None:
            return self._cake._layers_last_index[self._cake_order]
        else:
            return None

    def add_equation(self, equation):
        """Add an equation object to the layer.

        Parameters
        ----------
        equation: ~equation.Equation
            Equation object to add to the layer.
        """
        self.equations.append(equation)
        equation._layer = self
        equation.field._layer = self

    @property
    def fields(self):
        """list(~field.Field): Returns the list of dynamical fields of the layer, i.e. the fields whose time
        evolution is provided by the partial differential equations of the layer."""
        fields_list = list()
        for eq in self.equations:
            fields_list.append(eq.field)
        return fields_list

    @property
    def parameters(self):
        """list(~parameter.Parameter): Returns the list of parameters of the layer, i.e. the explicit parameters
        appearing in the partial differential equations of the layer."""
        parameters_list = list()
        for eq in self.equations:
            for param in eq.parameters:
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
        """list(~sympy.core.symbol.Symbol): List of parameter's symbols present in the layer
        partial differential equations."""
        return [p.symbol for p in self.parameters]

    @property
    def _fields_layer_tensor_extent(self):
        extent = dict()
        n = 1
        for field in self.fields:
            ni = n + field.state.__len__()
            extent[field] = (n, ni)
            n = ni
        return extent

    @property
    def ndim(self):
        """int: Dimension of the full ordinary differential equations system of the layer, resulting from
        the Galerkin expansion."""
        dim = 0
        for field in self.fields:
            dim += field.state.__len__()
        return dim

    @property
    def number_of_equations(self):
        """int: Number of partial differential equations in the layer."""
        return self.equations.__len__()

    @property
    def maximum_rank(self):
        """int: Maximum over the ranks of the equations in the layer."""
        max_rank = 0
        for eq in self.equations:
            max_rank = max(max_rank, eq.maximum_rank)
        return max_rank

    def compute_inner_products(self, numerical=True, timeout=None, num_threads=None):
        """Compute the inner products tensors, either symbolic or numerical ones, of all the terms
        of the layer equations, including the left-hand side term.
        Computations are parallelized on multiple CPUs.

        Parameters
        ----------
        numerical: bool, optional
            Whether to compute numerical or symbolic inner products.
            Default to `True` (numerical inner products as output).
        timeout: int or bool or None, optional
            TODO
        num_threads: None or int, optional
            Number of CPUs to use in parallel for the computations. If `None`, use all the CPUs available.
            Default to `None`.
        """
        for field, eq in zip(self.fields, self.equations):
            eq.lhs_term.compute_inner_products(field.basis, numerical=numerical, timeout=timeout, num_threads=num_threads)
            for term in eq.terms:
                term.compute_inner_products(field.basis, numerical=numerical, timeout=timeout, num_threads=num_threads)

    def compute_tensor(self, numerical=True, compute_inner_products=False, compute_inner_products_kwargs=None,
                       substitutions=None, basis_subs=False, parameters_subs=None):
        """Compute the tensor of the symbolic or numerical representation of the ordinary differential
        equations tendencies of the layer.
        Results are stored in the :attr:`~Layer.tensor` attribute.


        Parameters
        ----------
        numerical: bool, optional
            Whether to compute the numerical or the symbolic tensor.
            Default to `True` (numerical tensor as output).
        compute_inner_products: bool, optional
            Whether the inner products tensors of the layer equations' terms must be compute first.
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

        if compute_inner_products_kwargs is None:
            compute_inner_products_kwargs = dict()

        if compute_inner_products:
            self.compute_inner_products(numerical=numerical, **compute_inner_products_kwargs)

        if self._cake is not None:
            shape = tuple([self.ndim + 1] + [self._cake.ndim + 1] * (self.maximum_rank - 1))
        else:
            shape = tuple([self.ndim + 1] * self.maximum_rank)

        if numerical:

            self.tensor = sp.zeros(shape, dtype=np.float64, format='dok')
            lhs_mat = np.zeros((self.ndim+1, self.ndim+1))
            lhs_order = 1
            for field, eq in zip(self.fields, self.equations):
                ndim = field.state.__len__()
                try:
                    lhs_mat[lhs_order:lhs_order + ndim, lhs_order:lhs_order + ndim] = np.linalg.inv(eq.lhs_term.inner_products.todense())
                except LinAlgError:
                    raise LinAlgError(f'The left-hand side of the equation {eq} is not invertible with the provided basis.')
                for equation_term in eq.terms:
                    slices = [slice(lhs_order, lhs_order + ndim)]
                    for term in equation_term.terms:
                        term_field = term.field
                        if term_field.dynamical:
                            if self._cake is not None:
                                term_extent = self._cake.fields_tensor_extent[term_field]
                            elif term_field in self.fields:
                                term_extent = self._fields_layer_tensor_extent[term_field]
                            else:
                                raise AttributeError(f'Field {term_field} provided in equation {eq} cannot be found in the cake or in the layer.')
                            slices.append(slice(*term_extent))
                        else:
                            slices.append(0)
                    zeros = [0 for _ in range(equation_term.rank, len(self.tensor.shape))]
                    args = slices+zeros
                    if isinstance(equation_term, ConstantTerm):
                        increment = equation_term.field.parameters.astype(float)
                    else:
                        increment = equation_term.inner_products.todense()
                        if isinstance(equation_term, ProductOfTerms):
                            contract = dict()
                            for i, t in enumerate(equation_term.terms):
                                if isinstance(t.field, (ParameterField, FunctionField)):
                                    params = t.field.parameters.astype(float)
                                    contract[i] = params
                            if contract:
                                for i in sorted(list(contract.keys()), reverse=True):
                                    params = contract[i]
                                    increment = np.tensordot(increment, params, ((i+1,), (0,)))
                                    args[i+1] = 0
                        elif hasattr(equation_term, 'field'):
                            if isinstance(equation_term.field, (ParameterField, FunctionField)):
                                params = equation_term.field.parameters.astype(float)
                                increment = np.tensordot(increment, params, ((1,), (0,)))
                                args[1] = 0
                    args = tuple(args)
                    self.tensor[args] = self.tensor[args] + increment
                lhs_order += ndim
            self.tensor = sp.COO(np.tensordot(lhs_mat, self.tensor.to_coo(), 1))

        else:
            b_subs = list()
            if substitutions is None:
                substitutions = list()
            if parameters_subs is not None:
                p_subs = [(param.symbol, float(param)) for param in parameters_subs]
                # TODO: Seems to not allow ParameterField to be substituted. To check.
            else:
                p_subs = list()
            self.tensor = MutableSparseNDimArray(iterable={}, shape=shape)
            lhs_mat = MutableSparseMatrix(sympy_zeros(self.ndim + 1, self.ndim + 1))
            lhs_order = 1
            for field, eq in zip(self.fields, self.equations):
                bsb = field.basis.substitutions
                if basis_subs:
                    for sbsb in bsb:
                        for obsb in b_subs:
                            if sbsb[0] == obsb[0]:
                                break
                        else:
                            b_subs.append(sbsb)
                ndim = field.state.__len__()
                try:
                    lhs_mat[lhs_order:lhs_order + ndim, lhs_order:lhs_order + ndim] = eq.lhs_term.inner_products.inverse().simplify()
                except NonInvertibleMatrixError:
                    raise NonInvertibleMatrixError(f'The left-hand side of the equation {eq} is not invertible with the provided basis.')
                for equation_term in eq.terms:
                    slices = [slice(lhs_order, lhs_order + ndim)]
                    for term in equation_term.terms:
                        term_field = term.field
                        if term_field.dynamical:
                            if self._cake is not None:
                                term_extent = self._cake.fields_tensor_extent[term_field]
                            elif term_field in self.fields:
                                term_extent = self._fields_layer_tensor_extent[term_field]
                            else:
                                raise AttributeError(f'Field {term_field} provided in equation {eq} cannot be found in the cake or in the layer.')
                            slices.append(slice(*term_extent))
                        else:
                            slices.append(0)
                    zeros = [0 for _ in range(equation_term.rank, len(self.tensor.shape))]
                    args = slices+zeros
                    if isinstance(equation_term, ConstantTerm):
                        term_symbol_list = list()
                        term_symbol_list.append(list(equation_term.field.symbols))
                        increment = ImmutableMatrix(term_symbol_list).reshape(len(term_symbol_list[0]), 1)
                    else:
                        increment = equation_term.inner_products
                        if isinstance(equation_term, ProductOfTerms):
                            contract = dict()
                            for i, t in enumerate(equation_term.terms):
                                if isinstance(t.field, (ParameterField, FunctionField)):
                                    term_symbol_list = list()
                                    term_symbol_list.append(list(t.field.symbols))
                                    params = ImmutableMatrix(term_symbol_list).reshape(len(term_symbol_list[0]), 1)
                                    contract[i] = params
                            if contract:
                                for i in sorted(list(contract.keys()), reverse=True):
                                    params = contract[i]
                                    increment = symbolic_tensordot(increment, params, ((i+1,), (0,)))
                                    args[i+1] = 0
                                    iargs = list()
                                    for j in increment.shape[:-1]:
                                        iargs.append(slice(j))
                                    iargs.append(0)
                                    increment = increment[tuple(iargs)]
                        elif hasattr(equation_term, 'field'):
                            if isinstance(equation_term.field, (ParameterField, FunctionField)):
                                term_symbol_list = list()
                                term_symbol_list.append(list(equation_term.field.symbols))
                                params = ImmutableMatrix(term_symbol_list).reshape(len(term_symbol_list[0]), 1)
                                increment = symbolic_tensordot(increment, params, ((1,), (0,)))
                                args[1] = 0
                                iargs = list()
                                for j in increment.shape[:-1]:
                                    iargs.append(slice(j))
                                iargs.append(0)
                                increment = increment[tuple(iargs)]
                    args = tuple(args)
                    if increment.is_Matrix:
                        increment = ImmutableSparseNDimArray(increment)
                    self.tensor[args] = self.tensor[args] + increment
                lhs_order += ndim
            self.tensor = (ImmutableSparseNDimArray(symbolic_tensordot(lhs_mat, self.tensor, 1))
                           .subs(b_subs).subs(p_subs).subs(substitutions))

    def to_latex(self, enclose_lhs=True, drop_first_lhs_char=True, drop_first_rhs_char=False):
        """Generate the LaTeX strings representing the layer's equations mathematically.

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
        list(str)
            The LaTeX strings representing the layer's equations.
        """

        latex_string_list = list()

        for eq in self.equations:
            latex_string_list.append(eq.to_latex(enclose_lhs=enclose_lhs,
                                                 drop_first_lhs_char=drop_first_lhs_char,
                                                 drop_first_rhs_char=drop_first_rhs_char
                                                 )
                                     )

        return latex_string_list

    def show_latex(self, enclose_lhs=True, drop_first_lhs_char=True, drop_first_rhs_char=False):
        """Show the LaTeX string representing the layer's equations mathematically rendered in a window.

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

        latex_string_list = self.to_latex(enclose_lhs=enclose_lhs,
                                          drop_first_lhs_char=drop_first_lhs_char,
                                          drop_first_rhs_char=drop_first_rhs_char
                                          )

        plt.figure(figsize=(8, self.number_of_equations))
        plt.axis('off')
        for i, s in enumerate(latex_string_list):
            plt.text(-0.1, (self.number_of_equations - i) / (self.number_of_equations + 1), '$%s$' % s)
        plt.show()
