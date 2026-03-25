
"""
    Field definition module
    =======================

    This module defines spatial fields in the models.

    Description of the classes
    --------------------------

    * :class:`Field`: Class defining the spatial fields.
    * :class:`ParameterField`: Class defining static spatial field that can be viewed as models' parameters.

"""

import os
from pebble import ProcessPool as Pool
from multiprocessing import cpu_count
import numpy as np
from sympy import Symbol, Function
from sympy import ImmutableSparseMatrix

from layercake.variables.variable import Variable, VariablesArray
from layercake.variables.parameter import ParametersArray
from layercake.utils.parallel import parallel_integration
from layercake.utils.integration import integration
from layercake.utils.symbolic_tensor import remove_dic_zeros


class Field(Variable):
    """ Class defining the spatial fields in the models.

    Parameters
    ----------
    name: str
        Name of the field.
    symbol: ~sympy.core.symbol.Symbol
        A |Sympy| symbol to represent the field in symbolic expressions.
    basis: SymbolicBasis
        A symbolic basis of functions on which the Galerkin expansion of the field is performed.
    inner_product_definition: InnerProductDefinition
        Inner product definition object used to compute the inner products between the elements of the basis.
    units: str, optional
        The units of the variable.
        Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
        Empty by default.
    latex: str, optional
        Latex string representing the variable.
        Empty by default.
    state: VariablesArray
        Field state array: array containing the coefficient of the field's Galerkin expansion.
    state_kwargs: dict
        Used to create the field state array if `state` is not a :class:`VariableArray`.
        Passed to the :class:`VariableArray` constructor.

    Attributes
    ----------
    name: str
        Name of the field.
    symbol: ~sympy.core.symbol.Symbol
        The |Sympy| symbol representing the field in symbolic expressions.
    basis: SymbolicBasis
        The symbolic basis of functions on which the Galerkin expansion of the field is performed.
    inner_product_definition: InnerProductDefinition
        The inner product definition object used to compute the inner products between the elements of the basis.
    units: str, optional
        The units of the variable.
        Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
    latex: str, optional
        Latex string representing the variable.
    state: VariablesArray
        Field state array: array containing the coefficient of the field's Galerkin expansion.
    coordinate_system: CoordinateSystem
        Coordinate system on which the basis of functions and the inner product are defined.
    function: ~sympy.core.expr.Expr
        A Sympy symbolic representation of the field as a function of space and time.

    """

    def __init__(self, name, symbol, basis, inner_product_definition=None, units=None, latex=None, state=None, **state_kwargs):

        _t = Symbol('t')

        Variable.__init__(self, name, symbol, units, latex, True)

        self.basis = basis
        self.coordinate_system = basis.coordinate_system
        self.inner_product_definition = inner_product_definition
        self.function = Function(symbol)(_t, *self.coordinate_system.coordinates_symbol_as_list)
        if 'dynamical' not in state_kwargs:
            state_kwargs['dynamical'] = True
        if state is None:
            self.state = VariablesArray(np.zeros(len(self.basis)), name, symbol, latex=latex, **state_kwargs)
        elif isinstance(state, VariablesArray):
            self.state = state
        else:
            self.state = VariablesArray(state, name, symbol, latex=latex, **state_kwargs)
        self._layer = None
        self._cake = None
        self._equation = None

    def __str__(self):
        return self.name + ' (symbol: ' + str(self.symbol) + ',  units: ' + self.units + ', state: ' + str(self.state) + ' )'

    def __repr__(self):
        return self.__str__()


class ParameterField(Variable):
    """ Class defining a static spatial fields in the models.
    Can be viewed as a model parameter.

    Parameters
    ----------
    name: str
        Name of the field.
    symbol: ~sympy.core.symbol.Symbol
        A |Sympy| symbol to represent the field in symbolic expressions.
    parameters_array: ParametersArray or ~numpy.ndarray
        Array containing the coefficients of the field Galerkin expansion.
    basis: SymbolicBasis
        A symbolic basis of functions on which the Galerkin expansion of the field is performed.
    inner_product_definition: InnerProductDefinition
        Inner product definition object used to compute the inner products between the elements of the basis.
    units: str, optional
        The units of the variable.
        Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
        Empty by default.
    latex: str, optional
        Latex string representing the variable.
        Empty by default.
    parameters_array_kwargs: dict, optional
        Used to create the field state if `parameters_array` is not a :class:`ParametersArray` object.
        Passed to the :class:`ParametersArray` class constructor.

    Attributes
    ----------
    name: str
        Name of the field.
    symbol: ~sympy.core.symbol.Symbol
        The |Sympy| symbol representing the field in symbolic expressions.
    basis: SymbolicBasis
        The symbolic basis of functions on which the Galerkin expansion of the field is performed.
    inner_product_definition: InnerProductDefinition
        The inner product definition object used to compute the inner products between the elements of the basis.
    units: str
        The units of the variable.
        Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
    latex: str, optional
        Latex string representing the variable.
    parameters: ParametersArray
        Array containing the coefficients of the field Galerkin expansion.

    """

    def __init__(self, name, symbol, parameters_array, basis,
                 inner_product_definition=None, units="", latex=None, **parameters_array_kwargs):

        self.basis = basis
        self.inner_product_definition = inner_product_definition
        if isinstance(parameters_array, ParametersArray):
            self.parameters = parameters_array
            self.units = parameters_array.units
        else:
            self.units = units
            if isinstance(symbol, Symbol):
                symbol_name = symbol.name
            else:
                symbol_name = symbol
            symbols = [Symbol(f'{symbol_name}_{i}') for i in range(len(parameters_array))]
            if isinstance(parameters_array, np.ndarray):
                symbols = np.array(symbols, dtype=object)
            self.parameters = ParametersArray(parameters_array, units=units, symbols=symbols, **parameters_array_kwargs)

        Variable.__init__(self, name, symbol, self.parameters.units, latex)
        if self.parameters.__len__() != len(basis):
            raise ValueError('The number of parameters provided does not match the number of modes in the provided basis.')

    @property
    def dimensional_values(self):
        """float: Returns the dimensional value."""
        return self.parameters.dimensional_values

    @property
    def nondimensional_values(self):
        """float: Returns the nondimensional value."""
        return self.parameters.nondimensional_values

    @property
    def symbols(self):
        """~numpy.ndarray(~sympy.core.symbol.Symbol): Returns the symbol of the parameters in the array."""
        return self.parameters.symbols

    @property
    def symbolic_expressions(self):
        """~numpy.ndarray(~sympy.core.expr.Expr): Returns the symbolic expressions of the parameters in the array."""
        return self.parameters.symbolic_expressions

    @property
    def input_dimensional(self):
        """bool: Indicate if the provided value is dimensional or not."""
        return self.parameters.input_dimensional

    @property
    def return_dimensional(self):
        """bool: Indicate if the returned value is dimensional or not."""
        return self.parameters.return_dimensional

    @property
    def descriptions(self):
        """~numpy.ndarray(str): Description of the parameters in the array."""
        return self.parameters.descriptions

    def __str__(self):
        return self.name + ' (symbol: ' + str(self.symbol) + ',  units: ' + self.units + ', parameters: ' + str(self.parameters) + ' )'

    def __repr__(self):
        return self.__str__()


class FunctionField(Variable):
    """ Class defining a static spatial fields in the models, specified as a function of the model's coordinates.

    Parameters
    ----------
    name: str
        Name of the field.
    symbol: ~sympy.core.symbol.Symbol
        Symbol representing the field.
    symbolic_expression: ~sympy.core.expr.Expr
        A |Sympy| expression to represent the field mathematical expression in symbolic expressions.
    basis: SymbolicBasis
        A symbolic basis of functions on which the Galerkin expansion of the field is performed.
    expression_parameters: None or list(~parameter.Parameter), optional
        List of parameters appearing in the symbolic expression.
        If `None`, assumes that no parameters are appearing there.
    inner_product_definition: InnerProductDefinition or None, optional
        Inner product definition object used to compute the inner products between the elements of the basis.
        If `None`, the Galerkin expansion will not be computed, and then this field can only be used in symbolic expressions.
    units: str, optional
        The units of the variable.
        Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
        Empty by default.
    latex: str, optional
        Latex string representing the variable.
        Empty by default.
    extra_substitutions: list(tuple), optional
        List of 2-tuples containing extra symbolic substitutions to be made at the end of the integral computation.
        The 2-tuples contain first a |Sympy|  expression and then the value to substitute.
    force_substitution: bool, optional
        Force the substitution by the numerical values of the parameters arrays, even in the symbolic case.
        Default to `False`.
    force_symbolic_substitution: bool, optional
        Force the substitution by the symbolic expressions resulting from the projection of the function field onto
        the provided `basis`. Only relevant when working in the symbolic case.
        Is superseded by the `force_substitution` argument, i.e. the latter should be set to `False` for this parameter to work.
        Default to `False`.
    **parameters_array_kwargs: dict, optional
        Used to create the field state :class:`ParametersArray` object.
        Passed to the :class:`ParametersArray` class constructor.

    Attributes
    ----------
    name: str
        Name of the field.
    symbol: ~sympy.core.symbol.Symbol
        Symbol representing the field.
    symbolic_expression: ~sympy.core.expr.Expr
        A |Sympy| expression to represent the field mathematical expression in symbolic expressions.
    basis: SymbolicBasis
        The symbolic basis of functions on which the Galerkin expansion of the field is performed.
    inner_product_definition: InnerProductDefinition
        The inner product definition object used to compute the inner products between the elements of the basis.
    units: str
        The units of the variable.
        Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
    latex: str, optional
        Latex string representing the variable.
    parameters: ParametersArray
        Array containing the coefficients of the field Galerkin expansion, computed from the specified basis.
    expression_parameters: None or list(~parameter.Parameter), optional
        List of parameters appearing in the symbolic expression.
        If `None`, assumes that no parameters are appearing there.
    force_substitution: bool, optional
        Force the substitution by the numerical values of the parameters arrays, even in the symbolic case.
    force_symbolic_substitution: bool, optional
        Force the substitution by the symbolic expressions resulting from the projection of the function field onto
        the provided `basis`.

    """

    def __init__(self, name, symbol, symbolic_expression, basis, expression_parameters=None,
                 inner_product_definition=None, units="", latex=None, extra_substitutions=None,
                 force_substitution=False, force_symbolic_substitution=False, **parameters_array_kwargs):

        self.basis = basis
        self.inner_product_definition = inner_product_definition
        self.symbolic_expression = symbolic_expression
        self.expression_parameters = expression_parameters
        self.units = units
        self.force_substitution = force_substitution
        self.force_symbolic_substitution = force_symbolic_substitution

        Variable.__init__(self, name, symbol, self.units, latex)

        self.parameters = None
        if self.inner_product_definition is not None:
            self._compute_expansion(timeout=True, num_threads=None, extra_substitutions=extra_substitutions, **parameters_array_kwargs)
            if self.parameters.__len__() != len(basis):
                raise ValueError('The number of parameters provided does not match the number of modes in the provided basis.')

        self.symbolic_parameters = None
        if self.inner_product_definition is not None:
            self._compute_symbolic_expansion(timeout=None, num_threads=None, extra_substitutions=extra_substitutions)
            if self.symbolic_parameters.shape[1] != len(basis):
                raise ValueError('The number of parameters provided does not match the number of modes in the provided basis.')

    @property
    def dimensional_values(self):
        """float: Returns the dimensional value."""
        return self.parameters.dimensional_values

    @property
    def nondimensional_values(self):
        """float: Returns the nondimensional value."""
        return self.parameters.nondimensional_values

    @property
    def symbols(self):
        """~numpy.ndarray(~sympy.core.symbol.Symbol): Returns the symbol of the parameters in the array."""
        if isinstance(self.symbol, Symbol):
            return self.parameters.symbols
        else:
            return self.parameters

    @property
    def symbolic_expressions(self):
        """~numpy.ndarray(~sympy.core.expr.Expr): Returns the symbolic expressions of the parameters in the array."""
        if isinstance(self.symbol, Symbol):
            return self.parameters.symbolic_expressions
        else:
            return self.parameters

    @property
    def numerical_expression(self):
        """~sympy.core.expr.Expr: The numeric expression of the function, i.e. with parameters replaced by their numerical value."""
        substitutions = list()
        if self.expression_parameters is None:
            expr_parameters = list()
        else:
            expr_parameters = self.expression_parameters
        for param in expr_parameters:
            substitutions.append((param.symbol, float(param)))
        return self.symbolic_expression.subs(substitutions)

    @property
    def input_dimensional(self):
        """bool: Indicate if the provided value is dimensional or not."""
        return self.parameters.input_dimensional

    @property
    def return_dimensional(self):
        """bool: Indicate if the returned value is dimensional or not."""
        return self.parameters.return_dimensional

    @property
    def descriptions(self):
        """~numpy.ndarray(str): Description of the parameters in the array."""
        return self.parameters.descriptions

    def __str__(self):
        return self.name + ' (symbol: ' + str(self.symbol) + ',  units: ' + self.units + ', parameters: ' + str(self.parameters) + ' )'

    def __repr__(self):
        return self.__str__()

    def _compute_expansion(self, timeout=None, num_threads=None, extra_substitutions=None, **parameters_array_kwargs):
        """Compute the Galerkin expansion and store the result.

        Parameters
        ----------
        timeout: None or bool or int
            Control the switch from symbolic to numerical integration.
            In the end, all results are converted to numerical expressions, but
            by default, `parallel_integration` workers will try first to integrate
            |Sympy| expressions symbolically. However, a fallback to numerical integration can be enforced.
            The options are:

            * `None`: This is the "full-symbolic" mode. No timeout will be applied, and the switch to numerical integration will never happen.
              Can result in very long and improbable computation time.
            * `True`: This is the "full-numerical" mode. Symbolic computations do not occur, and the workers try directly to integrate
              numerically.
            * `False`: Same as `None`.
            * An integer: defines a timeout after which, if a symbolic integration have not completed, the worker switch to the
              numerical integration.
        num_threads: None or int, optional
            Number of CPUs to use in parallel for the computations. If `None`, use all the CPUs available.
            Default to `None`.
        extra_substitutions: list(tuple)
            List of 2-tuples containing extra symbolic substitutions to be made at the end of the integral computation.
            The 2-tuples contain first a |Sympy|  expression and then the value to substitute.
        **parameters_array_kwargs: dict, optional
            Used to create the field state :class:`ParametersArray` object.
            Passed to the :class:`ParametersArray` class constructor.

        """
        if num_threads is None:
            num_threads = cpu_count()

        substitutions = self.basis.substitutions
        if self.expression_parameters is None:
            expr_parameters = list()
        else:
            expr_parameters = self.expression_parameters
        for param in expr_parameters:
            substitutions.append((param.symbol, float(param)))
        if extra_substitutions is not None:
            substitutions += extra_substitutions

        args_list = [(idx, self.inner_product_definition.inner_product,
                      (self.basis[idx].subs(substitutions), self.symbolic_expression.subs(substitutions)))
                     for idx in range(len(self.basis))]

        if 'LAYERCAKE_PARALLEL_INTEGRATION' not in os.environ:
            parallel_integrations = True
        else:
            if os.environ['LAYERCAKE_PARALLEL_INTEGRATION'] == 'none':
                parallel_integrations = False
            else:
                parallel_integrations = True

        if parallel_integrations:
            with Pool(max_workers=num_threads) as pool:
                output = parallel_integration(pool, args_list, substitutions, None, timeout,
                                              symbolic_int=False, permute=False)
        else:
            output = integration(args_list, substitutions, None, symbolic_int=False, permute=False)

        res = np.zeros(len(self.basis), dtype=float)
        for i in output:
            if isinstance(output[i], (float, int)):
                res[i] = output[i]
            else:
                res[i] = output[i].subs(substitutions)

        symbol_name = self.symbol.name
        symbols_list = [Symbol(f'{symbol_name}_{i}') for i in range(len(self.basis))]
        symbols_list = np.array(symbols_list, dtype=object)
        self.parameters = ParametersArray(res, units=self.units, symbols=symbols_list, **parameters_array_kwargs)

    def _compute_symbolic_expansion(self, timeout=None, num_threads=None, basis_subs=False, extra_substitutions=None):
        """Compute the Galerkin expansion and store the result.

        Parameters
        ----------
        timeout: None or bool or int
            Control the switch from symbolic to numerical integration.
            In the end, all results are converted to numerical expressions, but
            by default, `parallel_integration` workers will try first to integrate
            |Sympy| expressions symbolically. However, a fallback to numerical integration can be enforced.
            The options are:

            * `None`: This is the "full-symbolic" mode. No timeout will be applied, and the switch to numerical integration will never happen.
              Can result in very long and improbable computation time.
            * `True`: This is the "full-numerical" mode. Symbolic computations do not occur, and the workers try directly to integrate
              numerically.
            * `False`: Same as `None`.
            * An integer: defines a timeout after which, if a symbolic integration have not completed, the worker switch to the
              numerical integration.
        num_threads: None or int, optional
            Number of CPUs to use in parallel for the computations. If `None`, use all the CPUs available.
            Default to `None`.
        basis_subs: bool, optional
            Whether to substitute the parameters appearing in the definition of the basis of functions by
            their numerical value.
            Default to `False`.
        extra_substitutions: list(tuple)
            List of 2-tuples containing extra symbolic substitutions to be made at the end of the integral computation.
            The 2-tuples contain first a |Sympy|  expression and then the value to substitute.

        """
        if num_threads is None:
            num_threads = cpu_count()

        if basis_subs:
            substitutions = self.basis.substitutions
        else:
            substitutions = list()
        if extra_substitutions is not None:
            substitutions += extra_substitutions

        args_list = [(idx, self.inner_product_definition.inner_product,
                      (self.basis[idx].subs(substitutions), self.symbolic_expression.subs(substitutions)))
                     for idx in range(len(self.basis))]

        if 'LAYERCAKE_PARALLEL_INTEGRATION' not in os.environ:
            parallel_integrations = True
        else:
            if os.environ['LAYERCAKE_PARALLEL_INTEGRATION'] == 'none':
                parallel_integrations = False
            else:
                parallel_integrations = True

        if parallel_integrations:
            with Pool(max_workers=num_threads) as pool:
                output = parallel_integration(pool, args_list, substitutions, None, timeout,
                                              symbolic_int=True, permute=False)
        else:
            output = integration(args_list, substitutions, None, symbolic_int=True, permute=False)

        output = remove_dic_zeros(output)
        mat_output = {(0, idx): v for idx, v in output.items()}
        self.symbolic_parameters = ImmutableSparseMatrix(1, len(self.basis), mat_output)
        self.symbolic_parameters = self.symbolic_parameters.subs(substitutions)


if __name__ == "__main__":
    from layercake import Parameter
    from layercake.basis.spherical_harmonics import SphericalHarmonicsBasis
    from sympy import symbols, sin, cos
    from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition

    _R = symbols('R')
    R = Parameter(1., symbol=_R)
    parameters = [R]
    basis = SphericalHarmonicsBasis(parameters, {'M': 4})
    s = StandardSymbolicInnerProductDefinition(basis.coordinate_system, optimizer=None)

    cs = s.coordinate_system
    phi = cs.coordinates_symbol_as_list[1]

    p = u'ψ'
    psi = Field("psi", p, basis, s, units="[m^2][s^-2]", latex=r'\psi')
    ff = FunctionField('ff', 's_phi', sin(phi), basis, inner_product_definition=s, latex=r'\sin \phi')
    ffc = FunctionField('ffc', 'c_phi', R * cos(phi), basis, inner_product_definition=s, latex=r'\cos \phi')
