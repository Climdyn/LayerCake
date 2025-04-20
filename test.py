from layercake.basis.planar_fourier import contiguous_basin_basis, contiguous_channel_basis
from sympy import symbols
from layercake.variables.parameter import ScalingParameter
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition

_n = symbols('n')

n = ScalingParameter(1.3, symbol=_n)

parameters = {'n': n}

b = contiguous_basin_basis(2, 2, parameters)

s = StandardSymbolicInnerProductDefinition(coordinate_system=b.coordinate_system)

