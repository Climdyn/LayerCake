
from layercake.variables.variable import Variable
from sympy import Symbol, Function


class Field(Variable):

    def __init__(self, name, symbol, basis, inner_product_definition=None, units=None, latex=None):

        _t = Symbol('t')

        self.name = name
        self.basis = basis
        self.coordinate_system = basis.coordinate_system
        self.inner_product_definition = inner_product_definition
        if isinstance(symbol, str):
            self.symbol = Symbol(symbol)
        else:
            self.symbol = symbol
        self.function = Function(symbol)(_t, *self.coordinate_system.coordinates_symbol_as_list)
        if units is None:
            self.units = ""
        else:
            self.units = units

        if latex is None:
            self.latex = name
        else:
            self.latex = latex

    def __str__(self):
        return self.name + ' (symbol: ' + str(self.symbol) + ',  units: ' + self.units + ')'

    def __repr__(self):
        return self.__str__()

