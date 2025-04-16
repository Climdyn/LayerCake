
from layercake.variables.coordinates import Coordinate
from sympy import symbols


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


class PlanarCartesianCoordinateSystem(CoordinateSystem):

    def __init__(self, extent):

        xs, ys = symbols('x y')
        x = Coordinate("x", xs, extent=extent[0])
        y = Coordinate("y", ys, extent=extent[1])
        CoordinateSystem.__init__(self, coordinates=[x, y], name="Planar Cartesian Coordinate System")
