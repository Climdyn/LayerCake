import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from layercake.basis.planar_fourier import contiguous_channel_basis
from sympy import symbols, Symbol
from layercake.variables.parameter import Parameter
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition
from layercake.variables.field import Field, ParameterField
from layercake.arithmetic.terms.jacobian import vorticity_advection, Jacobian
from layercake.arithmetic.terms.operators import OperatorTerm
from layercake.arithmetic.terms.operations import AdditionOfTerms
from layercake.arithmetic.terms.linear import LinearTerm
from layercake.arithmetic.equation import Equation
from layercake.arithmetic.symbolic.operators import Laplacian, D
from layercake.bakery.layers import Layer
from layercake.bakery.cake import Cake


# Defining the domain
ns = symbols('n')
n = Parameter(1.5, symbol=ns)
parameters = {'n': n}
b = contiguous_channel_basis(2, 2, parameters)
s = StandardSymbolicInnerProductDefinition(coordinate_system=b.coordinate_system)
# coordinates
x = b.coordinate_system.coordinates_symbol_as_list[0]
y = b.coordinate_system.coordinates_symbol_as_list[1]

# Defining the fields
p = u'ψ'
psi = Field("psi", p, b, s, units="[m^2][s^-2]", latex=r'\psi_{\rm a}')
tt = u'θ'
theta = Field("theta", tt, b, s, units="[m^2][s^-2]", latex=r'\theta_{\rm a}')

p_o = u'ψ_o'
psi_o = Field("psi_o", p, b, s, units="[m^2][s^-2]", latex=r'\psi_{\rm o}')
tto = u'T_o'
T_o = Field("T_o", tt, b, s, units="[m^2][s^-2]", latex=r'T_{\rm o}')

# --------------------------------
#
#   Barotropic field equation
#
# --------------------------------

# Defining the equation and LHS
# Laplacian
vorticity = OperatorTerm(psi, Laplacian, b.coordinate_system)
barotropic_equation = Equation(psi, lhs_term=vorticity)

# Defining the advection term
advection_term1 = vorticity_advection(psi, psi, b.coordinate_system, sign=-1)
advection_term2 = vorticity_advection(theta, theta, b.coordinate_system, sign=-1)

barotropic_equation.add_rhs_terms(advection_term1)
barotropic_equation.add_rhs_terms(advection_term2)

# adding an orographic term
g = 0.5
gamma = symbols(u'γ')
gammap = Parameter(g, symbol=gamma)
hh = np.zeros(len(b))
hh[1] = 0.2
h = ParameterField('h', u'h', hh, b, s)

orographic_term1 = Jacobian(psi, h, b.coordinate_system, sign=-1, prefactors=(gammap, gammap))
orographic_term2 = Jacobian(theta, h, b.coordinate_system, sign=1, prefactors=(gammap, gammap))

barotropic_equation.add_rhs_terms(orographic_term1)
barotropic_equation.add_rhs_terms(orographic_term2)

# adding the beta term
betaa = symbols(u'β')
beta = Parameter(0.20964969238375256, symbol=betaa)
betaterm = OperatorTerm(psi, D, x, prefactor=beta, sign=-1)

barotropic_equation.add_rhs_term(betaterm)

# adding the atmospheric friction
kdd = symbols('k_d')
kd = Parameter(0.05, symbol=kdd)
friction = OperatorTerm(psi, Laplacian, b.coordinate_system, prefactor=kd, sign=-1)
barotropic_equation.add_rhs_term(friction)

ofriction = OperatorTerm(theta, Laplacian, b.coordinate_system, prefactor=kd)
barotropic_equation.add_rhs_term(ofriction)

ocfriction = OperatorTerm(psi_o, Laplacian, b.coordinate_system, prefactor=kd)
barotropic_equation.add_rhs_term(ocfriction)




# --------------------------------
#
#   Baroclinic field equation
#
# --------------------------------

# Defining the equation and LHS
# Laplacian
vorticity = OperatorTerm(theta, Laplacian, b.coordinate_system)
a_symbol = symbols('a')
a = Parameter(2 / 0.2, symbol=a_symbol)

lin_lhs = LinearTerm(theta, prefactor=a, sign=-1)
lhs = AdditionOfTerms(lin_lhs, vorticity)
baroclinic_equation = Equation(theta, lhs_term=lhs)

# Defining the advection terms
advection_term1 = vorticity_advection(psi, theta, b.coordinate_system, sign=-1)
advection_term2 = vorticity_advection(theta, psi, b.coordinate_system, sign=-1)

baroclinic_equation.add_rhs_terms(advection_term1)
baroclinic_equation.add_rhs_terms(advection_term2)


# adding an orographic term

orographic_term1 = Jacobian(psi, h, b.coordinate_system, sign=1, prefactors=(gammap, gammap))
orographic_term2 = Jacobian(theta, h, b.coordinate_system, sign=-1, prefactors=(gammap, gammap))

baroclinic_equation.add_rhs_terms(orographic_term1)
baroclinic_equation.add_rhs_terms(orographic_term2)

# adding the beta term
betaa = symbols(u'β')
beta = Parameter(0.20964969238375256, symbol=betaa)
betaterm = OperatorTerm(theta, D, x, prefactor=beta, sign=-1)

baroclinic_equation.add_rhs_term(betaterm)


# adding the atmospheric friction
friction = OperatorTerm(psi, Laplacian, b.coordinate_system, prefactor=kd, sign=1)
baroclinic_equation.add_rhs_term(friction)

ofriction = OperatorTerm(theta, Laplacian, b.coordinate_system, prefactor=kd, sign=-1)
baroclinic_equation.add_rhs_term(ofriction)

ocfriction = OperatorTerm(psi_o, Laplacian, b.coordinate_system, prefactor=kd, sign=-1)
baroclinic_equation.add_rhs_term(ocfriction)

# adding the friction with the ground
kddp = symbols('k_dp')
kdp = Parameter(2 * 0.01, symbol=kddp)
ground_friction = OperatorTerm(theta, Laplacian, b.coordinate_system, prefactor=kdp, sign=-1)
baroclinic_equation.add_rhs_term(ground_friction)

# adding jacobian from thermal wind relation

thermal = Jacobian(psi, theta, b.coordinate_system, prefactors=(a, a))
baroclinic_equation.add_rhs_terms(thermal)


# adding heat exchange scheme

la1 = Symbol('λ1')
la1p = Parameter(, symbol=la1)

la1t = LinearTerm(theta, prefactor=la1p, sign=1)

la2 = Symbol('λ2')
la2p = Parameter(, symbol=la2)

la2t = LinearTerm(T_o, prefactor=la2p, sign=-1)

baroclinic_equation.add_rhs_terms((la1t, la2t))

# adding the insolation

Cas = Symbol('Ca')
Cav = np.zeros(len(b))
Cav[0] = 0.1
Ca = ParameterField('Ca', Cas, Cav, b, s)
Cat = LinearTerm(Ca, sign=-1)

baroclinic_equation.add_rhs_term(Cat)



# adding friction with the ocean
dd = symbols('d')
d = Parameter(0.05, symbol=dd)

# --------------------------------
#
#   Constructing the layer
#
# --------------------------------

layer = Layer()
layer.add_equation(barotropic_equation)
layer.add_equation(baroclinic_equation)


# --------------------------------
#
#   Constructing the cake
#
# --------------------------------

cake = Cake()
cake.add_layer(layer)

# --------------------------------
#
#   Computing the tendencies
#
# --------------------------------

# computing the tensor
cake.compute_tensor(True, True
                    )
# computing the tendencies
f, Df = cake.compute_tendencies()

# integrating
ic = np.random.rand(cake.ndim) * 0.1
res = solve_ivp(f, (0., 20000.), ic)

ic = res.y[:, -1]
res = solve_ivp(f, (0., 20000.), ic)

# plotting
plt.plot(res.y.T)
plt.show()

