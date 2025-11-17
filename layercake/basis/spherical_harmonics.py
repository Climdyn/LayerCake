
from sympy import assoc_legendre, exp, I, sin, pi

from layercake.basis.base import SymbolicBasis
from layercake.variables.systems import SphericalCoordinateSystem

from layercake.variables.parameter import Parameter


class SphericalHarmonicsBasis(SymbolicBasis):

    def __init__(self, parameters, truncation_parameter, truncation='T'):

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

                    mode_eq = assoc_legendre(n, m, sin(phi)) * exp(I * m * (llambda + pi))

                    if mode_eq is not None:
                        self.functions.append(mode_eq)

        else:
            raise NotImplementedError("Only triangular ('T') truncation is implemented for the moment.")


# if __name__ == "__main__":

