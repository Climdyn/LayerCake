import numpy as np
from sympy import Symbol, Function
from layercake.variables.variable import Variable, VariablesArray
from layercake.variables.parameter import ParametersArray


class Field(Variable):

    def __init__(self, name, symbol, basis, inner_product_definition=None, units=None, latex=None, state=None, **state_kwargs):

        _t = Symbol('t')

        Variable.__init__(self, name, symbol, units, latex, True)

        self.basis = basis
        self.coordinate_system = basis.coordinate_system
        self.inner_product_definition = inner_product_definition
        self.function = Function(symbol)(_t, *self.coordinate_system.coordinates_symbol_as_list)
        if state is None:
            self.state = VariablesArray(np.zeros(len(self.basis)), name, symbol, latex=latex, **state_kwargs)
        elif isinstance(state, VariablesArray):
            self.state = state
        else:
            self.state = VariablesArray(state, name, symbol, latex=latex, **state_kwargs)
        # self._layer = None
        # self._cake = None
        # self._equation = None

    def __str__(self):
        return self.name + ' (symbol: ' + str(self.symbol) + ',  units: ' + self.units + ', state: ' + str(self.state) + ' )'

    def __repr__(self):
        return self.__str__()


class ParameterField(Variable):

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

