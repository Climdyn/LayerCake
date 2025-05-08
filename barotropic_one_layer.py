import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from layercake.basis.planar_fourier import contiguous_channel_basis
from sympy import symbols
from layercake.variables.parameter import ScalingParameter
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition
from layercake.variables.field import Field, ParameterField
from layercake.arithmetic.terms.operations import ProductOfTerms
from layercake.arithmetic.terms.operators import OperatorTerm, ComposedOperatorsTerm
from layercake.arithmetic.equation import Equation
from layercake.arithmetic.symbolic.operators import Laplacian, D
from layercake.bakery.layers import Layer
from layercake.bakery.cake import Cake


# Defining the domain
_n = symbols('n')
n = ScalingParameter(1.3, symbol=_n)
parameters = {'n': n}
b = contiguous_channel_basis(2, 2, parameters)
s = StandardSymbolicInnerProductDefinition(coordinate_system=b.coordinate_system)
# coordinates
x = b.coordinate_system.coordinates_symbol_as_list[0]
y = b.coordinate_system.coordinates_symbol_as_list[1]

# Defining the field
p = u'ψ'
psi = Field("psi", p, b, s, units="[m^2][s^-2]", latex=r'\psi')

# Defining the equation and LHS
# Laplacian
lapo = OperatorTerm(psi, Laplacian, b.coordinate_system)
e = Equation(psi, lhs_term=lapo, inner_product_definition=s)

# Defining the Jacobian
dxpsi = OperatorTerm(psi, D, x)
dypsi = OperatorTerm(psi, D, y)

dxlapopsi = ComposedOperatorsTerm(psi, (D, Laplacian), (x, b.coordinate_system))

dylapopsi = ComposedOperatorsTerm(psi, (D, Laplacian), (y, b.coordinate_system))

jacobian1 = ProductOfTerms(dxpsi, dylapopsi, sign=-1)
jacobian2 = ProductOfTerms(dypsi, dxlapopsi)

e.add_rhs_terms([jacobian1, jacobian2])

# adding an orographic term
g = 0.1
gamma = symbols(u'γ')
gammap = ScalingParameter(g, symbol=gamma)
hh = np.zeros(len(b))
hh[1] = 1.
h = ParameterField('h', u'h', hh, b, s)

hdxpsi = OperatorTerm(psi, D, x, prefactor=gammap)
hdyh = OperatorTerm(h, D, y)

hdypsi = OperatorTerm(psi, D, y, prefactor=gammap)
hdxh = OperatorTerm(h, D, x)

hjacobian1 = ProductOfTerms(hdxpsi, hdyh, sign=-1)
hjacobian2 = ProductOfTerms(hdypsi, hdxh)

e.add_rhs_terms([hjacobian1, hjacobian2])

# adding the beta term
betaa = symbols(u'β')
beta = ScalingParameter(0.20964969238375256, symbol=betaa)
betaterm = OperatorTerm(psi, D, x, prefactor=beta, sign=-1)

e.add_rhs_term(betaterm)

# adding a friction
kdd = symbols('k_d')
kd = ScalingParameter(0.05, symbol=kdd)
friction = OperatorTerm(psi, Laplacian, b.coordinate_system, prefactor=kd, sign=-1)
e.add_rhs_term(friction)

# adding an interaction with a background streamfunction
Cdd = symbols('C')
Cd = ScalingParameter(0.05, symbol=Cdd)
rr = np.zeros(len(b))
rr[0] = 0.3
C = ParameterField('eta', u'η', rr, b, s)
CT = OperatorTerm(C, Laplacian, b.coordinate_system, prefactor=Cd)

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
ic = np.random.rand(cake.ndim) * 0.1
res = solve_ivp(f, (0., 1000.), ic)

# plotting
plt.plot(res.y.T)
plt.show()

