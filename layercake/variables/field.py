
"""
    Field definition module
    =======================

    This module defines spatial fields in the models.

    Description of the classes
    --------------------------

    * :class:`Field`: Class defining the spatial fields.
    * :class:`ParameterField`: Class defining static spatial field that can be viewed as models' parameters.

"""

import numpy as np
from sympy import Symbol, Function
from layercake.variables.variable import Variable, VariablesArray
from layercake.variables.parameter import ParametersArray


class Field(Variable):
    """ Class defining the spatial fields in the models.

    Parameters
    ----------
    name: str
        Name of the field.
    symbol: ~sympy.core.symbol.Symbol
        A `Sympy`_ symbol to represent the field in symbolic expressions.
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
        The `Sympy`_ symbol representing the field in symbolic expressions.
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
        A `Sympy`_ symbol to represent the field in symbolic expressions.
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
    parameters_array: ParametersArray
        Array containing the coefficients of the field Galerkin expansion.
    parameters_array_kwargs: dict
        Used to create the field state if `parameters_array` is not a :class:`ParametersArray` object.
        Passed to the :class:`ParametersArray` class constructor.

    Attributes
    ----------
    name: str
        Name of the field.
    symbol: ~sympy.core.symbol.Symbol
        The `Sympy`_ symbol representing the field in symbolic expressions.
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
