import numpy as np

from layercake.basis.planar_fourier import contiguous_basin_basis, contiguous_channel_basis
from sympy import symbols
from layercake.variables.parameter import ScalingParameter
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition
from layercake.variables.field import Field, ParameterField
from layercake.arithmetic.terms.linear import LinearTerm
from layercake.arithmetic.terms.constant import ConstantTerm
from layercake.arithmetic.terms.operations import ProductOfTerms
from layercake.arithmetic.terms.operators import OperatorTerm, ComposedOperatorsTerm
from layercake.arithmetic.equation import Equation
from layercake.variables.systems import SphericalCoordinateSystem
from layercake.arithmetic.symbolic.operators import Nabla, Laplacian, Divergence, D
from layercake.bakery.layers import Layer
from layercake.bakery.cake import Cake


# Defining a parameter equal to -1

aa = symbols('a')
a = ScalingParameter(- 1, symbol=aa)

# Defining the domain
_n = symbols('n')
n = ScalingParameter(1.3, symbol=_n)
parameters = {'n': n}
b = contiguous_channel_basis(1, 2, parameters)
s = StandardSymbolicInnerProductDefinition(coordinate_system=b.coordinate_system)

# Defining the field
p = u'ψ'
psi = Field("psi", p, b, s, units="[m^2][s^-2]", latex=r'\psi')

# Defining the equation and LHS
# Laplacian
lapo = OperatorTerm(psi, Laplacian, b.coordinate_system)
e = Equation(psi, lhs_term=lapo, inner_product_definition=s)

# Defining the Jacobian
dxpsi = OperatorTerm(psi, D, b.coordinate_system.coordinates_symbol_as_list[0], prefactor=a)
dypsi = OperatorTerm(psi, D, b.coordinate_system.coordinates_symbol_as_list[1]) #, prefactor=a)

dxlapopsi = ComposedOperatorsTerm(psi, (D, Laplacian), (b.coordinate_system.coordinates_symbol_as_list[0],
                                                    b.coordinate_system))

dylapopsi = ComposedOperatorsTerm(psi, (D, Laplacian), (b.coordinate_system.coordinates_symbol_as_list[1],
                                                        b.coordinate_system))

jacobian1 = ProductOfTerms(dxpsi, dylapopsi)
jacobian2 = ProductOfTerms(dypsi, dxlapopsi)

e.add_rhs_terms([jacobian1, jacobian2])

# adding the beta term
betaa = symbols(u'β')
beta = ScalingParameter(-1.25, symbol=betaa)
betaterm = OperatorTerm(psi, D, b.coordinate_system.coordinates_symbol_as_list[0], prefactor=beta)

e.add_rhs_term(betaterm)

# adding a friction
kdd = symbols('k_d')
kd = ScalingParameter(-0.1, symbol=kdd)
friction = OperatorTerm(psi, Laplacian, b.coordinate_system, prefactor=kd)
e.add_rhs_term(friction)

# adding a constant term
rr = np.zeros(len(b))
rr[0] = - 0.1 * 0.95
rr[3] = + 0.1 * 4 * 0.76095
C = ParameterField('R', u'R', rr, b, s)
CT = ConstantTerm(C)

e.add_rhs_term(CT)

# constructing the layer
layer = Layer()
layer.add_equation(e)

# constructing the cake
cake = Cake()
cake.add_layer(layer)

# computing the tensor
cake.compute_tensor(True, True
                    )
# computing the tendencies
f, Df = cake.compute_tendencies()

# integrating
from scipy.integrate import solve_ivp
ic= np.zeros(6)+0.1
res = solve_ivp(f,(0.,1000.), ic)

# plotting
import matplotlib.pyplot as plt
plt.plot(res.y.T)
plt.show()

