from layercake.basis.planar_fourier import contiguous_basin_basis, contiguous_channel_basis
from sympy import symbols
from layercake.variables.parameter import ScalingParameter
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition
from layercake.variables.field import Field
from layercake.arithmetic.terms.linear import LinearTerm
from layercake.arithmetic.equation import Equation

_n = symbols('n')

n = ScalingParameter(1.3, symbol=_n)

parameters = {'n': n}

b = contiguous_basin_basis(2, 2, parameters)

s = StandardSymbolicInnerProductDefinition(coordinate_system=b.coordinate_system)


p = u'ψ'

psi = Field("psi", p, units="[m^2][s^-2]", latex=r'\psi', coordinate_system=b.coordinate_system)

aa = symbols('a')
a = ScalingParameter(- 2, symbol=aa)

l = LinearTerm(psi, s, a)

e = Equation(psi)
e.add_term(l)


