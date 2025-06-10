import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from layercake.basis.planar_fourier import contiguous_channel_basis, contiguous_basin_basis
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

# Setting some parameters
##########################

# Characteristic length scale (L_y / pi)
L_symbol = Symbol('L')
L = Parameter(1591549.4309189534, symbol=L_symbol, units='[m]')

# Domain aspect ratio
n_symbol = symbols('n')
n = Parameter(1.5, symbol=n_symbol)

# Coriolis parameter at the middle of the domain
f0_symbol = Symbol('f_0')
f0 = Parameter(1.032e-4, symbol=f0_symbol, units='[s^-1]')

# Pressure difference between the two atmospheric layers
deltap_symbol = Symbol('Δp')
deltap = Parameter(5.e4, symbol=deltap_symbol, units='[Pa]')

# Static stability of the atmosphere
sigma_symbol = symbols('σ')
sigma = Parameter(2.1581898457499433e-06, symbol=sigma_symbol, units='[m^2][s^-2][Pa^-2]')

# Meridional gradient of the Coriolis parameter at phi_0
beta_symbol = Symbol(u'β')
beta = Parameter(1.3594204385792041e-11, symbol=beta_symbol, units='[m^-1][s^-1]')

# atmosphere bottom friction coefficient
kd_symbol = Symbol('k_d')
kd = Parameter(1.032e-05, symbol=kd_symbol, units='[s^-1]')

# Atmosphere internal friction coefficient
kdp_symbol = Symbol("k_d'")
kdp = Parameter(1.032e-06, symbol=kdp_symbol, units='[s^-1]')

# Friction between the ocean and the atmosphere
d_symbol = symbols('d')
d = Parameter(0.05, symbol=d_symbol)

# Friction of the ocean with the bottom
r_symbol = symbols('r')
r = Parameter(1.e-7, symbol=r_symbol, units='[s^-1]')

# Physical constants
# gas constant of dry air
rr_symbol = Symbol('R')
rr = Parameter(287.058, symbol=rr_symbol, units='[J][kg^-1][K^-1]')
# Stefan-Boltzmann constant
sb_symbol = Symbol('σ_B')
sb = Parameter(5.67e-08, symbol=sb_symbol, units='[J][m^-2][s^-1][K^-4]')

# Sensible + turbulent heat exchange between ocean/ground and atmosphere
hlambda_symbol = Symbol('λ')
hlambda = Parameter(15.06, symbol=hlambda_symbol, units='[W][m^-2][K^-1]')

# Stationary solution for the 0-th order atmospheric temperature
T0a_symbol = Symbol('T_{a, 0}')
T0a = Parameter(289.3, symbol=T0a_symbol, units='[K]')

# Stationary solution for the 0-th order atmospheric temperature
T0o_symbol = Symbol('T_{o, 0}')
T0o = Parameter(301.46, symbol=T0o_symbol, units='[K]')

# Specific heat capacity of the ocean
gamma_o_symbol = Symbol('γ_o')
gamma_o = Parameter(560000000.0, symbol=gamma_o_symbol, units='[J][m^-2][K^-1]')

# Specific heat capacity of the atmosphere
gamma_a_symbol = Symbol('γ_a')
gamma_a = Parameter(10000000.0, symbol=gamma_a_symbol, units='[J][m^-2][K^-1]')

# Depth of the water layer of the ocean
h_symbol = Symbol('h')
h = Parameter(136.5, symbol=h_symbol, units='[m]')

# Emissivity coefficient for the grey-body atmosphere
eps_symbol = Symbol('ε')
eps = Parameter(0.7, symbol=eps_symbol)

# Defining the domain
######################

parameters = {'n': n}
atmospheric_basis = contiguous_channel_basis(2, 2, parameters)
oceanic_basis = contiguous_basin_basis(2, 2, parameters)
s = StandardSymbolicInnerProductDefinition(coordinate_system=b.coordinate_system)
# coordinates
x = atmospheric_basis.coordinate_system.coordinates_symbol_as_list[0]
y = atmospheric_basis.coordinate_system.coordinates_symbol_as_list[1]

# Derived (non-dimensional) parameters
#######################################

sigma_nondim = Parameter((sigma * deltap ** 2) / (L ** 2 * f0 ** 2), symbol=sigma_symbol, units='')
beta_nondim = Parameter(beta * L / f0, symbol=beta_symbol, units='')
a_symbol = symbols('a')
a = Parameter(2 / sigma_nondim, symbol=a_symbol)
kd_deriv = Parameter(0.5 * kd / f0, symbol=kd_symbol)
kdp_deriv = Parameter(2 * kdp / f0, symbol=kdp_symbol)

# Defining the fields
#######################
p = u'ψ_a'
psi_a = Field("psi_a", p, atmospheric_basis, s, units="[m^2][s^-2]", latex=r'\psi_{\rm a}')
tt = u'θ_a'
theta_a = Field("theta_a", tt, atmospheric_basis, s, units="[m^2][s^-2]", latex=r'\theta_{\rm a}')

p_o = u'ψ_o'
psi_o = Field("psi_o", p, atmospheric_basis, s, units="[m^2][s^-2]", latex=r'\psi_{\rm o}')
tto = u'δT_o'
deltaT_o = Field("deltaT_o", tt, atmospheric_basis, s, units="[m^2][s^-2]", latex=r'\delta T_{\rm o}')

# --------------------------------------------------------
#
#   Barotropic atmospheric field equation
#
# --------------------------------------------------------

# Defining the equation and LHS
# Laplacian
vorticity = OperatorTerm(psi_a, Laplacian, atmospheric_basis.coordinate_system)
barotropic_equation = Equation(psi_a, lhs_term=vorticity)

# Defining the advection term
advection_term1 = vorticity_advection(psi_a, psi_a, atmospheric_basis.coordinate_system, sign=-1)
advection_term2 = vorticity_advection(theta_a, theta_a, atmospheric_basis.coordinate_system, sign=-1)

barotropic_equation.add_rhs_terms(advection_term1)
barotropic_equation.add_rhs_terms(advection_term2)

# adding the beta term
beta_term = OperatorTerm(psi_a, D, x, prefactor=beta_nondim, sign=-1)
barotropic_equation.add_rhs_term(beta_term)

# adding the atmospheric friction
friction = OperatorTerm(psi_a, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv, sign=-1)
barotropic_equation.add_rhs_term(friction)

ofriction = OperatorTerm(theta_a, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv)
barotropic_equation.add_rhs_term(ofriction)

ocfriction = OperatorTerm(psi_o, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv)
barotropic_equation.add_rhs_term(ocfriction)


# --------------------------------------------------------
#
#   Baroclinic atmospheric field equation
#
# --------------------------------------------------------

# Defining the equation and LHS
# Laplacian
vorticity = OperatorTerm(theta_a, Laplacian, atmospheric_basis.coordinate_system)

lin_lhs = LinearTerm(theta_a, prefactor=a, sign=-1)
lhs = AdditionOfTerms(lin_lhs, vorticity)
baroclinic_equation = Equation(theta_a, lhs_term=lhs)

# Defining the advection terms
advection_term1 = vorticity_advection(psi_a, theta_a, atmospheric_basis.coordinate_system, sign=-1)
advection_term2 = vorticity_advection(theta_a, psi_a, atmospheric_basis.coordinate_system, sign=-1)

baroclinic_equation.add_rhs_terms(advection_term1)
baroclinic_equation.add_rhs_terms(advection_term2)

# adding the beta term
beta_term = OperatorTerm(theta_a, D, x, prefactor=beta_nondim, sign=-1)
baroclinic_equation.add_rhs_term(beta_term)

# adding the friction with the ocean
friction = OperatorTerm(psi_a, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv, sign=1)
baroclinic_equation.add_rhs_term(friction)

ofriction = OperatorTerm(theta_a, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv, sign=-1)
baroclinic_equation.add_rhs_term(ofriction)

ocfriction = OperatorTerm(psi_o, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv, sign=-1)
baroclinic_equation.add_rhs_term(ocfriction)

# adding the atmospheric friction
ground_friction = OperatorTerm(theta_a, Laplacian, atmospheric_basis.coordinate_system, prefactor=kdp_deriv, sign=-1)
baroclinic_equation.add_rhs_term(ground_friction)

# adding jacobian from thermal wind relation

thermal = Jacobian(psi_a, theta_a, atmospheric_basis.coordinate_system, prefactors=(a, a))
baroclinic_equation.add_rhs_terms(thermal)

# adding heat exchange scheme

la1 = Symbol('λ1')
la1p = Parameter(, symbol=la1)

la1t = LinearTerm(theta_a, prefactor=la1p, sign=1)

la2 = Symbol('λ2')
la2p = Parameter(, symbol=la2)

la2t = LinearTerm(deltaT_o, prefactor=la2p, sign=-1)

baroclinic_equation.add_rhs_terms((la1t, la2t))

# adding the insolation

Cas = Symbol('Ca')
Cav = np.zeros(len(atmospheric_basis))
Cav[0] = 0.1
Ca = ParameterField('Ca', Cas, Cav, atmospheric_basis, s)
Cat = LinearTerm(Ca, sign=-1)

baroclinic_equation.add_rhs_term(Cat)


# --------------------------------------------------------
#
#   Shallow-water oceanic field equation
#
# --------------------------------------------------------

# Defining the equation and LHS
# Laplacian
vorticity = OperatorTerm(psi_o, Laplacian, oceanic_basis.coordinate_system)
G_symbol = symbols('G')
G = Parameter(1., symbol=G_symbol)

lin_lhs = LinearTerm(psi_o, prefactor=G)
lhs = AdditionOfTerms(lin_lhs, vorticity)
oceanic_equation = Equation(psi_o, lhs_term=lhs)

# Defining the advection term
advection_term = vorticity_advection(psi_o, psi_o, oceanic_basis.coordinate_system, sign=-1)
oceanic_equation.add_rhs_terms(advection_term)

# adding the beta term
beta_term = OperatorTerm(psi_o, D, x, prefactor=beta_nondim, sign=-1)
oceanic_equation.add_rhs_term(beta_term)

# adding friction with the ocean
friction = OperatorTerm(psi_a, Laplacian, oceanic_basis.coordinate_system, prefactor=d, sign=1)
oceanic_equation.add_rhs_term(friction)

ofriction = OperatorTerm(theta_a, Laplacian, oceanic_basis.coordinate_system, prefactor=d, sign=-1)
oceanic_equation.add_rhs_term(ofriction)

ocfriction = OperatorTerm(psi_o, Laplacian, oceanic_basis.coordinate_system, prefactor=d, sign=-1)
oceanic_equation.add_rhs_term(ocfriction)

# adding friction at the bottom of the ocean
bfriction = OperatorTerm(psi_o, Laplacian, oceanic_basis.coordinate_system, prefactor=r, sign=-1)
oceanic_equation.add_rhs_term(bfriction)


# --------------------------------------------------------
#
#   Oceanic temperature field equation
#
# --------------------------------------------------------

# Defining the equation and LHS
# Laplacian

lhs = LinearTerm(deltaT_o)
ocean_temperature_equation = Equation(theta_a, lhs_term=lhs)

# Defining the advection term
advection_term = Jacobian(psi_o, deltaT_o, oceanic_basis.coordinate_system, sign=-1)
ocean_temperature_equation.add_rhs_terms(advection_term)

# adding heat exchange scheme
lo3 = Symbol('λ3')
lo3p = Parameter(, symbol=lo3)

lo3t = LinearTerm(deltaT_o, prefactor=lo3p, sign=-1)

lo4 = Symbol('λ4')
lo4p = Parameter(, symbol=lo4)

lo4t = LinearTerm(theta_a, prefactor=lo4p, sign=1)

ocean_temperature_equation.add_rhs_terms((lo3t, lo4t))

# adding the insolation

Cos = Symbol('Co')
Cov = np.zeros(len(atmospheric_basis))
Cov[0] = 0.1
Co = ParameterField('Co', Cos, Cov, atmospheric_basis, s)
Cot = LinearTerm(Co, sign=-1)

ocean_temperature_equation.add_rhs_term(Cot)

# --------------------------------
#
#   Constructing the layers
#
# --------------------------------

atmospheric_layer = Layer()
atmospheric_layer.add_equation(barotropic_equation)
atmospheric_layer.add_equation(baroclinic_equation)


# --------------------------------
#
#   Constructing the cake
#
# --------------------------------

cake = Cake()
cake.add_layer(atmospheric_layer)

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

