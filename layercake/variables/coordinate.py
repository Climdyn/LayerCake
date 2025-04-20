
from layercake.variables.variable import Variable


class Coordinate(Variable):
    """Class to define a physical coordinate in the system.

    Parameters
    ----------
    symbol: ~sympy.core.symbol.Symbol
        Sympy symbol of the coordinate
    extent: tuple(float)
        The natural extent of the coordinate.
    units: str, optional
        The units of the coordinate. Used to compute the conversion between dimensional and nondimensional
        value. Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.

    Attributes
    ----------
    symbol: ~sympy.core.symbol.Symbol
        Sympy symbol of the coordinate
    extent: tuple(float)
        The natural extent of the coordinate.

    Warning
    -------
    Coordinates with infinite extent are not currently supported.

    """

    def __init__(self, name, symbol, extent, units=None):

        Variable.__init__(self, name, symbol, units)
        self.extent = extent


