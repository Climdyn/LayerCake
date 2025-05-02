
from abc import ABC
from sympy import Symbol, Function


class Variable(ABC):

    def __init__(self, name, symbol, units=None, latex=None):

        self.name = name
        if isinstance(symbol, str):
            self.symbol = Symbol(symbol)
        else:
            self.symbol = symbol
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

