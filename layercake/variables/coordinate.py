
from layercake.variables.variable import Variable
from sympy import S


class Coordinate(Variable):
    """Class to define a physical coordinate in the system.

    Parameters
    ----------
    name: str
        Name of the coordinate.
    symbol: ~sympy.core.symbol.Symbol
        Sympy symbol of the coordinate
    extent: tuple(float)
        The natural extent of the coordinate.
    units: str, optional
        The units of the coordinate. Used to compute the conversion between dimensional and nondimensional
        value. Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.

    Attributes
    ----------
    name: str
        Name of the coordinate.
    symbol: ~sympy.core.symbol.Symbol
        Sympy symbol of the coordinate
    extent: tuple(float)
        The natural extent of the coordinate.
    units: str, optional
        The units of the coordinate. Used to compute the conversion between dimensional and nondimensional
        value. Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.

    Warning
    -------
    Coordinates with infinite extent are not currently supported.

    """

    def __init__(self, name, symbol, extent, infinitesimal_length=S.One, units=None):

        Variable.__init__(self, name, symbol, units)
        self.extent = extent
        self.infinitesimal_length = infinitesimal_length


