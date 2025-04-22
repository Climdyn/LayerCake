
from layercake.variables.variable import Variable
from sympy import Symbol, Function


class Field(Variable):

    def __init__(self, name, symbol, coordinate_system, units=None, latex=None):

        t = Symbol('t')

        self.name = name
        self.coordinate_system = coordinate_system
        self.symbol = Symbol(symbol, commutative=False)
        self.function = Function(symbol)(t, *self.coordinate_system.coordinates_symbol_as_list)
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

