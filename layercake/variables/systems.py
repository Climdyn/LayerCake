"""

    Coordinate systems definition module
    ====================================

    Coordinate systems defined on models' domains.

    Description of the classes
    --------------------------

    * :class:`CoordinateSystem`: Base class to define coordinate systems for the the models.
    * :class:`PlanarCartesianCoordinateSystem`: Cartesian coordinate system defined on a plane.
    * :class:`SphericalCoordinateSystem`: Coordinate system defined on a sphere.

"""

from layercake.variables.coordinate import Coordinate
from sympy import symbols, Symbol, pi, cos, S


class CoordinateSystem(object):

    def __init__(self, coordinates, name=""):
        """
        Base class to define a coordinate system.

        Parameters
        ----------
        coordinates: list(~variable.Coordinate)
            List of coordinates on which the basis is defined.
        name: str, optional
            Optional name for the coordinate system.
        """
        self.name = name
        self.coordinates = coordinates

    @property
    def coordinates_symbol(self):
        """dict(~sympy.core.symbol.Symbol): Symbols of the coordinates as a dictionary."""
        return {coo.name: coo.symbol for coo in self.coordinates}

    @property
    def coordinates_symbol_as_list(self):
        """list(~sympy.core.symbol.Symbol): Symbols of the coordinates as a list."""
        return [coo.symbol for coo in self.coordinates]

    @property
    def coordinates_name(self):
        """list(str): List of the coordinates names."""
        return [coo.name for coo in self.coordinates]

    @property
    def extent(self):
        """dict(tuple(float or ~sympy.core.expr.Expr or ~sympy.core.symbol.Symbol)): Dictionary of the coordinates extents."""
        return {coo.name: coo.extent for coo in self.coordinates}

    @property
    def infinitesimal_volume(self):
        """~sympy.core.expr.Expr: Infinitesimal volume spanned by the coordinates."""
        volume = S.One
        for coordinate in self.coordinates:
            volume = volume * coordinate.infinitesimal_length

        return volume


class PlanarCartesianCoordinateSystem(CoordinateSystem):

    def __init__(self, extent):
        """
        Class to define a planar Cartesian coordinate system.
        Coordinates are :math:`x` and :math:`y`.

        Parameters
        ----------
        extent: list(tuple(float or ~sympy.core.expr.Expr or ~sympy.core.symbol.Symbol))
            Defines the extent of the plane.
        """

        xs, ys = symbols('x y')
        x = Coordinate("x", xs, extent=extent[0])
        y = Coordinate("y", ys, extent=extent[1])
        CoordinateSystem.__init__(self, coordinates=[x, y], name="Planar Cartesian Coordinate System")


class SphericalCoordinateSystem(CoordinateSystem):
    """
    Class to define a coordinate system on the sphere.
    Coordinates are the azimuth angle (longitude) :math:`\\lambda` and the
    elevation angle (latitude) :math:`\\varphi`.

    Parameters
    ----------
    radius: Variable or Parameter
        The radius of the sphere
    extent: list(tuple(float or ~sympy.core.expr.Expr or ~sympy.core.symbol.Symbol)), optional
        Defines the extent of the coordinates on the sphere.
        If not provided, the coordinate system covers the whole sphere.

    """

    def __init__(self, radius, extent=None):

        R = radius.symbol

        if extent is None:
            extent = ((-pi, pi), (-pi / 2, pi / 2))

        llambdas = Symbol(u'λ')
        phis = Symbol(u'ϕ')

        llambda = Coordinate('lambda', llambdas, extent=extent[0], infinitesimal_length=R * cos(phis))
        phi = Coordinate("phi", phis, extent=extent[1], infinitesimal_length=R)
        CoordinateSystem.__init__(self, coordinates=[llambda, phi], name='Spherical Coordinate System')
