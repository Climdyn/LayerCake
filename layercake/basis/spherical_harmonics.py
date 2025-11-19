"""
    Spherical Harmonics Basis definition module
    ===========================================

    Classes defining `Spherical Harmonics`_ basis of functions on a plane.

    .. _Spherical Harmonics: https://en.wikipedia.org/wiki/Spherical_harmonics

"""
from sympy import assoc_legendre, exp, I, sin, cos, sqrt, pi
from math import factorial

from layercake.basis.base import SymbolicBasis
from layercake.variables.systems import SphericalCoordinateSystem


class SphericalHarmonicsBasis(SymbolicBasis):
    """ Complex or real spherical harmonics basis defined on a sphere
    with a given radius :math:`R`.

    Parameters
    ----------
    parameters: list(~parameter.Parameter)
        List holding the parameters appearing in the equations defining the basis.
    truncation_parameter: dict
        Dictionary of parameter associated with the specified truncature.
        For example, for the default triangular truncation, it expects an entry `'M'`
        in the dictionary, determining the level of truncation `TM`.
    complex: bool, optional
        Whether the spherical harmonics are defined using complex functions.
        Default to `False`.
    truncation: str, optional
        Type of truncation to use.
        Default to `"T"` for a triangular truncature.
    exclude_constant_term: bool, optional
        Whether the spherical harmonics corresponding to a constant should be discarded.
        Default to `True`.

    Attributes
    ----------
    substitutions: list(tuple)
        List of 2-tuples containing the substitutions to be made with the functions. The 2-tuples contain first
        a |Sympy|  expression and then the value to substitute.
    coordinate_system: ~systems.CoordinateSystem
        Coordinate system on which the basis is defined.
    parameters: list(~parameter.Parameter)
        Dictionary holding the parameters appearing in the equations defining the basis.

    """

    def __init__(self, parameters, truncation_parameter, complex=False, truncation='T', exclude_constant_term=True):

        for param in parameters:
            if str(param.symbol) == 'R':
                break
        else:
            raise ValueError("Parameter 'R' (sphere radius) should be present in the provided parameters")

        radius = float(param)
        coordinate_system = SphericalCoordinateSystem(param)
        SymbolicBasis.__init__(self, coordinate_system, parameters)

        self._R = param.symbol
        self.substitutions.append((self._R, radius))
        self._map_mn = dict()

        llambda = coordinate_system.coordinates_symbol['lambda']
        phi = coordinate_system.coordinates_symbol['phi']

        if truncation == 'T':
            M = truncation_parameter['M']

            for m in range(-M, M+1):

                for n in range(abs(m), M):

                    if complex:
                        if exclude_constant_term:
                            if n == 0 and m == 0:
                                continue
                        mode_eq = (sqrt(2) * pi * sqrt(((2 * n + 1)/(4 * pi)) * (factorial(n - m)/factorial(n + m)))
                                   * assoc_legendre(n, m, sin(phi)) * exp(I * m * (llambda)))
                    else:
                        if exclude_constant_term:
                            if n == 0 and m == 0:
                                continue
                        if m < 0:
                            mode_eq = (2 * pi * sqrt(((2 * n + 1)/(4 * pi)) * (factorial(n + m)/factorial(n - m)))
                                       * assoc_legendre(n, -m, sin(phi)) * sin(-m * llambda))
                        elif m == 0:
                            mode_eq = sqrt((2 * n + 1)/(4 * pi)) * assoc_legendre(n, 0, sin(phi))

                        else:
                            mode_eq = (2 * pi * sqrt(((2 * n + 1)/(4 * pi)) * (factorial(n - m)/factorial(n + m)))
                                       * assoc_legendre(n, m, sin(phi)) * cos(m * llambda))

                    if mode_eq is not None:
                        self.functions.append(mode_eq)
                        if n not in self._map_mn:
                            self._map_mn[n] = dict()
                        if m not in self._map_mn[n]:
                            self._map_mn[n][m] = len(self.functions) - 1

        else:
            raise NotImplementedError("Only triangular ('T') truncation is implemented for the moment.")

    def find_functions(self, n, m):
        """Function which returns the index of the basis function of given n, m indices in the basis list.

        Parameters
        ----------
        m, n: int
            Spectral indices of the sought function.

        Returns
        -------
        int:
            The index of the basis function in the list.
        """
        if n in self._map_mn:
            if m in self._map_mn[n]:
                return self._map_mn[n][m]

        raise ValueError(f'Basis function for indices n={n} and m={m} not found.')

    def set_parameters(self, parameters):
        """Setter for the parameters' dictionary.

        Attributes
        ----------
        parameters: list(~parameter.Parameter)
            List holding the parameters appearing in the equations defining the basis.
        """

        for param in parameters:
            if str(param.symbol) == 'R':
                break
        else:
            raise ValueError("Parameter 'R' (sphere radius) should be present in the provided parameters")

        radius = float(param)
        coordinate_system = SphericalCoordinateSystem(param)
        self.coordinate_system = coordinate_system

        self._R = param.symbol
        self.substitutions = list()
        self.substitutions.append((self._R, radius))


if __name__ == "__main__":
    from layercake import Parameter
    from layercake.basis.spherical_harmonics import SphericalHarmonicsBasis
    from sympy import symbols
    from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition
    _R = symbols('R')
    R = Parameter(1., symbol=_R)
    parameters = [R]
    basis = SphericalHarmonicsBasis(parameters, {'M': 10})  # , complex=True)
    s = StandardSymbolicInnerProductDefinition(basis.coordinate_system, optimizer='trig')  # , complex=True)
    sn = StandardSymbolicInnerProductDefinition(basis.coordinate_system, optimizer=None)  # , complex=True)

