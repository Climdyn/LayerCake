
"""

    Coordinate definition module
    ============================

    A coordinate is a :class:`Variable` used to define the models' domains, i.e. as a part of a :class:`CoordinatesSystem`,
    to uniquely determine and standardize the position of the points of the domain.

    Description of the classes
    --------------------------

    * :class:`CoordinateSystem`: Base class to define coordinate systems for the models.
    * :class:`PlanarCartesianCoordinateSystem`: Cartesian coordinate system defined on a plane.
    * :class:`SphericalCoordinateSystem`: Coordinate system defined on a sphere.

"""
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
    extent: tuple(float or ~sympy.core.expr.Expr or ~sympy.core.symbol.Symbol)
        2-tuple giving the natural extent of the coordinate, i.e. the lower and higher bounds of the coordinate's interval.
    units: str, optional
        The units of the coordinate. Used to compute the conversion between dimensional and nondimensional
        value. Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
    infinitesimal_length: float or ~sympy.core.expr.Expr or ~sympy.core.symbol.Symbol
        Infinitesimal length associated with the

    Attributes
    ----------
    name: str
        Name of the coordinate.
    symbol: ~sympy.core.symbol.Symbol
        Sympy symbol of the coordinate
    extent: tuple(float or ~sympy.core.expr.Expr or ~sympy.core.symbol.Symbol)
        2-tuple giving the natural extent of the coordinate, i.e. the lower and higher bounds of the coordinate's interval.
    units: str, optional
        The units of the coordinate. Used to compute the conversion between dimensional and nondimensional
        value. Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
    infinitesimal_length: float or ~sympy.core.expr.Expr or ~sympy.core.symbol.Symbol
        Infinitesimal length associated with the

    Warning
    -------
    Coordinates with infinite extent are not currently supported.

    """

    def __init__(self, name, symbol, extent, infinitesimal_length=S.One, units=None):

        Variable.__init__(self, name, symbol, units)
        self.extent = extent
        self.infinitesimal_length = infinitesimal_length
