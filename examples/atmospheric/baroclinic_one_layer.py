import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import symbols, Symbol

import sys
import os
if os.path.basename(os.getcwd()) == 'LayerCake':
    sys.path.extend([os.path.abspath('./')])
else:
    sys.path.extend([os.path.abspath('../..')])

# importing all that is needed to create the cake
from layercake import *

# importing specific modules to create the model basis of functions
from layercake.basis.planar_fourier import contiguous_channel_basis
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition


##############################################################################################
#
# This script defines the Charney & Strauss 1980 baroclinic QG model 2-½ layer model
# (https://doi.org/10.1175/1520-0469(1980)037%3C1157:FDIMEA%3E2.0.CO;2)
# in the Reinhold & Pierrehumbert 1982 configuration
# (https://doi.org/10.1175/1520-0493(1982)110%3C1105:DOWRQS%3E2.0.CO;2)
# in numerical mode and then integrates (run) it, and plots the resulting trajectory in 2D.
#
# Note that the model is a two-layer one (250 and 750 HPa levels),
# but reduced on a single layer at 500 hPa, with a baroclinic (ψ) and &
# barotropic (θ) streamfunction on that level. (Hence the name "2-½ layer model".)
#
##############################################################################################

# Setting some parameters
##########################

# Characteristic length scale (L_y / pi)
L_symbol = Symbol('L')
L = Parameter(1591549.4309189534, symbol=L_symbol, units='[m]')

# Domain aspect ratio
n_symbol = Symbol('n')
n = Parameter(1.3, symbol=n_symbol)

# Coriolis parameter at the middle of the domain
f0_symbol = Symbol('f_0')
f0 = Parameter(1.032e-4, symbol=f0_symbol, units='[s^-1]')

# Pressure difference between the two atmospheric layers
deltap_symbol = Symbol('Δp')
deltap = Parameter(5.e4, symbol=deltap_symbol, units='[Pa]')

# Static stability of the atmosphere
sigma_symbol = Symbol('σ')
sigma = Parameter(2.1581898457499433e-06, symbol=sigma_symbol, units='[m^2][s^-2][Pa^-2]')

# Meridional gradient of the Coriolis parameter at phi_0
beta_symbol = Symbol(u'β')
beta = Parameter(1.3594204385792041e-11, symbol=beta_symbol, units='[m^-1][s^-1]')

# Atmosphere bottom friction coefficient
kd_symbol = Symbol('k_d')
kd = Parameter(1.032e-05, symbol=kd_symbol, units='[s^-1]')

# Atmosphere internal friction coefficient
kdp_symbol = Symbol("k_d'")
kdp = Parameter(1.032e-06, symbol=kdp_symbol, units='[s^-1]')

# Newtonian cooling parameters
hd_symbol = symbols('hd')
hd = Parameter(4.644e-06, symbol=hd_symbol, units='[s^-1]')

# Defining the domain
######################

parameters = [n]
atmospheric_basis = contiguous_channel_basis(2, 2, parameters)

# creating a inner product definition with an optimizer for trigonometric functions
inner_products_definition = StandardSymbolicInnerProductDefinition(coordinate_system=atmospheric_basis.coordinate_system,
                                                                   optimizer='trig', kwargs={'conds': 'none'})
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
hd_deriv = Parameter(a * hd / f0, symbol=hd_symbol, units='')

# Orography (non-dimensional)

hh = np.zeros(len(atmospheric_basis))
hh[1] = 0.2
h = ParameterField('h', u'h', hh, atmospheric_basis, inner_products_definition)

# Equilibrium temperature
rr = np.zeros(len(atmospheric_basis))
rr[0] = 0.1
Tf = ParameterField('T', u'T', rr, atmospheric_basis, inner_products_definition)

# Defining the fields
#######################
p = u'ψ'
psi = Field("psi", p, atmospheric_basis, inner_products_definition, units="[m^2][s^-2]", latex=r'\psi')
tt = u'θ'
theta = Field("theta", tt, atmospheric_basis, inner_products_definition, units="[m^2][s^-2]", latex=r'\theta')


# --------------------------------
#
#   Barotropic field equation
#
# --------------------------------

# defining the LHS as the time derivative of the vorticity
vorticity = OperatorTerm(psi, Laplacian, atmospheric_basis.coordinate_system)
barotropic_equation = Equation(psi, lhs_term=vorticity)

# Defining the advection term
advection_term1 = vorticity_advection(psi, psi, atmospheric_basis.coordinate_system, sign=-1)
advection_term2 = vorticity_advection(theta, theta, atmospheric_basis.coordinate_system, sign=-1)

barotropic_equation.add_rhs_terms(advection_term1)
barotropic_equation.add_rhs_terms(advection_term2)

# adding an orographic term
g = 0.5  # must be divided by 2
gamma = symbols(u'γ')
gammap = Parameter(g, symbol=gamma)

orographic_term1 = Jacobian(psi, h, atmospheric_basis.coordinate_system, sign=-1, prefactors=(gammap, gammap))
orographic_term2 = Jacobian(theta, h, atmospheric_basis.coordinate_system, sign=1, prefactors=(gammap, gammap))

barotropic_equation.add_rhs_terms(orographic_term1)
barotropic_equation.add_rhs_terms(orographic_term2)

# adding the beta term
beta_term = OperatorTerm(psi, D, x, prefactor=beta_nondim, sign=-1)

barotropic_equation.add_rhs_term(beta_term)

# adding the friction with the ground
friction = OperatorTerm(psi, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv, sign=-1)
barotropic_equation.add_rhs_term(friction)

ofriction = OperatorTerm(theta, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv)
barotropic_equation.add_rhs_term(ofriction)

# --------------------------------
#
#   Baroclinic field equation
#
# --------------------------------

# defining the LHS
vorticity = OperatorTerm(theta, Laplacian, atmospheric_basis.coordinate_system)

lin_lhs = LinearTerm(theta, prefactor=a, sign=-1)
lhs = AdditionOfTerms(lin_lhs, vorticity)
baroclinic_equation = Equation(theta, lhs_term=lhs)

# Defining the advection terms
advection_term1 = vorticity_advection(psi, theta, atmospheric_basis.coordinate_system, sign=-1)
advection_term2 = vorticity_advection(theta, psi, atmospheric_basis.coordinate_system, sign=-1)

baroclinic_equation.add_rhs_terms(advection_term1)
baroclinic_equation.add_rhs_terms(advection_term2)

# adding an orographic term
orographic_term1 = Jacobian(psi, h, atmospheric_basis.coordinate_system, sign=1, prefactors=(gammap, gammap))
orographic_term2 = Jacobian(theta, h, atmospheric_basis.coordinate_system, sign=-1, prefactors=(gammap, gammap))

baroclinic_equation.add_rhs_terms(orographic_term1)
baroclinic_equation.add_rhs_terms(orographic_term2)

# adding the beta term
beta_term = OperatorTerm(theta, D, x, prefactor=beta_nondim, sign=-1)
baroclinic_equation.add_rhs_term(beta_term)


# adding the friction with the ground
friction = OperatorTerm(psi, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv, sign=1)
baroclinic_equation.add_rhs_term(friction)

ofriction = OperatorTerm(theta, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv, sign=-1)
baroclinic_equation.add_rhs_term(ofriction)


# adding the atmospheric friction
ground_friction = OperatorTerm(theta, Laplacian, atmospheric_basis.coordinate_system, prefactor=kdp_deriv, sign=-1)
baroclinic_equation.add_rhs_term(ground_friction)

# adding jacobian from thermal wind relation
thermal = Jacobian(psi, theta, atmospheric_basis.coordinate_system, prefactors=(a, a))
baroclinic_equation.add_rhs_terms(thermal)


# adding Newtonian cooling
equilibrium_temperature = LinearTerm(Tf, prefactor=hd_deriv, sign=-1)
newt = LinearTerm(theta, prefactor=hd_deriv, sign=1)

baroclinic_equation.add_rhs_terms((newt, equilibrium_temperature))

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
cake.compute_tensor(True, True, compute_inner_products_kwargs={'timeout': True}
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
