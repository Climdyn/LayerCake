
import numpy as np
from numpy.linalg import LinAlgError
import sparse as sp
from sympy import MutableSparseNDimArray, MutableSparseMatrix, ImmutableMatrix, ImmutableSparseNDimArray
from sympy.matrices.exceptions import NonInvertibleMatrixError
from layercake.arithmetic.terms.constant import ConstantTerm
from layercake.arithmetic.terms.operations import ProductOfTerms
from layercake.variables.field import ParameterField
from layercake.utils.symbolic_tensor import symbolic_tensordot


class Layer(object):

    def __init__(self):

        self.equations = list()
        self.tensor = None
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
        self.equations.append(equation)
        equation._layer = self
        equation.field._layer = self

    @property
    def fields(self):
        fields_list = list()
        for eq in self.equations:
            fields_list.append(eq.field)
        return fields_list

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
        dim = 0
        for field in self.fields:
            dim += field.state.__len__()
        return dim

    @property
    def number_of_equations(self):
        return self.equations.__len__()

    @property
    def maximum_rank(self):
        max_rank = 0
        for eq in self.equations:
            max_rank = max(max_rank, eq.maximum_rank)
        return max_rank

    def compute_inner_products(self, numerical=True):
        for field, eq in zip(self.fields, self.equations):
            eq.lhs_term.compute_inner_products(field.basis, numerical=numerical)
            for term in eq.terms:
                term.compute_inner_products(field.basis, numerical=numerical)

    def compute_tensor(self, numerical=True, compute_inner_products=False, substitutions=None):

        if compute_inner_products:
            self.compute_inner_products(numerical=numerical)

        if self._cake is not None:
            shape = tuple([self.ndim + 1] + [self._cake.ndim + 1] * (self.maximum_rank - 1))
        else:
            shape = tuple([self.ndim + 1] * self.maximum_rank)

        if numerical:

            self.tensor = sp.zeros(shape, dtype=np.float64, format='dok')
            lhs_mat = MutableSparseMatrix(np.zeros((self.ndim+1, self.ndim+1)))
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
                                if isinstance(t.field, ParameterField):
                                    params = t.field.parameters.astype(float)
                                    contract[i] = params
                            if contract:
                                for i in sorted(list(contract.keys()), reverse=True):
                                    params = contract[i]
                                    increment = np.tensordot(increment, params, ((i+1,), (0,)))
                                    args[i+1] = 0
                        elif hasattr(equation_term, 'field'):
                            if isinstance(equation_term.field, ParameterField):
                                params = equation_term.field.parameters.astype(float)
                                increment = np.tensordot(increment, params, ((1,), (0,)))
                                args[1] = 0
                    args = tuple(args)
                    self.tensor[args] = self.tensor[args] + increment
                lhs_order += ndim
            self.tensor = sp.COO(np.tensordot(lhs_mat, self.tensor.to_coo(), 1))

        else:
            if substitutions is None:
                substitutions = dict()
            self.tensor = MutableSparseNDimArray(iterable=[0.,], shape=shape)
            lhs_mat = MutableSparseMatrix((self.ndim+1, self.ndim+1))
            lhs_order = 1
            for field, eq in zip(self.fields, self.equations):
                ndim = field.state.__len__()
                try:
                    lhs_mat[lhs_order:lhs_order + ndim, lhs_order:lhs_order + ndim] = eq.lhs_term.inner_products.subs(substitutions).inverse().symplify()
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
                        increment = ImmutableMatrix(term_symbol_list, shape=(len(term_symbol_list), 1))
                    else:
                        increment = equation_term.inner_products
                        if isinstance(equation_term, ProductOfTerms):
                            contract = dict()
                            for i, t in enumerate(equation_term.terms):
                                if isinstance(t.field, ParameterField):
                                    term_symbol_list = list()
                                    term_symbol_list.append(list(t.field.symbols))
                                    params = ImmutableMatrix(term_symbol_list, shape=(len(term_symbol_list), 1))
                                    contract[i] = params
                            if contract:
                                for i in sorted(list(contract.keys()), reverse=True):
                                    params = contract[i]
                                    increment = symbolic_tensordot(increment, params, ((i+1,), (0,)))
                                    args[i+1] = 0
                        elif hasattr(equation_term, 'field'):
                            if isinstance(equation_term.field, ParameterField):
                                term_symbol_list = list()
                                term_symbol_list.append(list(equation_term.field.symbols))
                                params = ImmutableMatrix(term_symbol_list, shape=(len(term_symbol_list), 1))
                                increment = symbolic_tensordot(increment, params, ((1,), (0,)))
                                args[1] = 0
                    args = tuple(args)
                    self.tensor[args] = self.tensor[args] + increment
                lhs_order += ndim
            self.tensor = ImmutableSparseNDimArray(symbolic_tensordot(lhs_mat, self.tensor, 1))



