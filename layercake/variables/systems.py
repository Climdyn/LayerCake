
from layercake.variables.coordinate import Coordinate
from sympy import symbols, Symbol, pi, cos, S


class CoordinateSystem(object):

    def __init__(self, coordinates, name=""):
        """
        Class to define a coordinate system.

        Parameters
        ----------
        coordinates: list(~variable.Coordinate)
            List of coordinates on which the basis is defined.
        """
        self.name = name
        self.coordinates = coordinates

    @property
    def coordinates_symbol(self):
        return {coo.name: coo.symbol for coo in self.coordinates}

    @property
    def coordinates_symbol_as_list(self):
        return [coo.symbol for coo in self.coordinates]

    @property
    def coordinates_name(self):
        return [coo.name for coo in self.coordinates]

    @property
    def extent(self):
        return {coo.name: coo.extent for coo in self.coordinates}

    @property
    def infinitesimal_volume(self):
        volume = S.One
        for coordinate in self.coordinates:
            volume = volume * coordinate.infinitesimal_length

        return volume


class PlanarCartesianCoordinateSystem(CoordinateSystem):

    def __init__(self, extent):

        xs, ys = symbols('x y')
        x = Coordinate("x", xs, extent=extent[0])
        y = Coordinate("y", ys, extent=extent[1])
        CoordinateSystem.__init__(self, coordinates=[x, y], name="Planar Cartesian Coordinate System")


class SphericalCoordinateSystem(CoordinateSystem):

    def __init__(self, radius, extent=None):

        R = radius.symbol

        if extent is None:
            extent = ((-pi, pi), (-pi / 2, pi / 2))

        llambdas = Symbol(u'λ')
        phis = Symbol(u'ϕ')

        llambda = Coordinate('lambda', llambdas, extent=extent[0], infinitesimal_length=R * cos(phis))
        phi = Coordinate("phi", phis, extent=extent[1], infinitesimal_length=R)
        CoordinateSystem.__init__(self, coordinates=[llambda, phi], name='Spherical Coordinate System')
