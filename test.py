from layercake.basis.planar_fourier import contiguous_basin_basis, contiguous_channel_basis
from sympy import symbols
from layercake.variables.parameter import ScalingParameter
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition
from layercake.variables.field import Field
from layercake.arithmetic.terms.linear import LinearTerm
from layercake.arithmetic.terms.operators import OperatorTerm, ComposedOperatorsTerm
from layercake.arithmetic.equation import Equation
from layercake.variables.systems import SphericalCoordinateSystem
from layercake.utils.operators import Nabla, Laplacian, Divergence, D

_n = symbols('n')

n = ScalingParameter(1.3, symbol=_n)

parameters = {'n': n}

# b = contiguous_basin_basis(2, 2, parameters)
b = contiguous_channel_basis(2, 2, parameters)

s = StandardSymbolicInnerProductDefinition(coordinate_system=b.coordinate_system)


p = u'ψ'

psi = Field("psi", p, units="[m^2][s^-2]", latex=r'\psi', coordinate_system=b.coordinate_system)

aa = symbols('a')
a = ScalingParameter(- 2, symbol=aa)
x = symbols('x')

l = LinearTerm(psi, s) #, a)
d = OperatorTerm(psi, s, D, b.coordinate_system.coordinates_symbol_as_list[0]) #, a)

e = Equation(psi, lhs_term=LinearTerm, inner_product_definition=s)
e.add_rhs_term(l)

R = symbols('R')
r = ScalingParameter(1., symbol=R)

scs = SphericalCoordinateSystem(r)

nab = Nabla(scs)
lap = Laplacian(scs)
div = Divergence(scs)

lapo = OperatorTerm(psi, s, Laplacian, b.coordinate_system)
e.add_rhs_terms([lapo, d])

c = ComposedOperatorsTerm(psi, s, D, b.coordinate_system.coordinates_symbol_as_list[0], Laplacian, b.coordinate_system)

e.add_rhs_term(c)
