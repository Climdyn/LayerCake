
from sympy import assoc_legendre, exp, I, sin, cos, sqrt, pi
from math import factorial

from layercake.basis.base import SymbolicBasis
from layercake.variables.systems import SphericalCoordinateSystem

from layercake.variables.parameter import Parameter


class SphericalHarmonicsBasis(SymbolicBasis):

    def __init__(self, parameters, truncation_parameter, complex=False, truncation='T'):

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

        llambda = coordinate_system.coordinates_symbol['lambda']
        phi = coordinate_system.coordinates_symbol['phi']

        if truncation == 'T':
            M = truncation_parameter['M']

            for m in range(-M, M+1):

                for n in range(abs(m), M):

                    if complex:
                        mode_eq = sqrt(2) * pi * sqrt(((2 * n + 1)/(4 * pi)) * (factorial(n - m)/factorial(n + m))) * assoc_legendre(n, m, sin(phi)) * exp(I * m * (llambda))
                    else:
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

        else:
            raise NotImplementedError("Only triangular ('T') truncation is implemented for the moment.")


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

