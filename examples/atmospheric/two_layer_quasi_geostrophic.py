import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import Symbol

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
# This script defines a two-layer quasi-geostrophic model on a beta-plane with an orography
# and is derived from Vallis book:
#       
#  Vallis, G. K. (2006). Atmospheric and oceanic fluid dynamics. Cambridge University Press.
#  See equation 5.86, chapter 5, pp. 226 .
#
#  In principle, the model conserves the potential vorticity in each layer.
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

# Meridional gradient of the Coriolis parameter at phi_0
beta_symbol = Symbol(u'β')
beta = Parameter(1.3594204385792041e-11, symbol=beta_symbol, units='[m^-1][s^-1]')

# Height of the atmospheric layers
H1_symbol = Symbol('H_1')
H1 = Parameter(5.2e3, symbol=H1_symbol, units='[m]')

H2_symbol = Symbol('H_2')
H2 = Parameter(5.2e3, symbol=H2_symbol, units='[m]')

# Gravity
g = Parameter(9.81, symbol=Symbol("g"), units='[m][s^-2]')

# Reduced gravity
gp = Parameter(float(g) * 0.6487, symbol=Symbol("g'"), units='[m][s^-2]')


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

# Meridional gradient of the Coriolis parameter at phi_0
beta_nondim = Parameter(beta * L / f0, symbol=beta_symbol, units='')

# Cross-terms
alpha_1 = Parameter(f0 ** 2 * L ** 2 / (g * H1), symbol=Symbol("α_1"), units='')
alphap_1 = Parameter(f0 ** 2 * L ** 2 / (gp * H1), symbol=Symbol("α'_1"), units='')
dalpha_1 = Parameter(alpha_1 - alphap_1, symbol=Symbol("Δα_1"), units='')
alphap_2 = Parameter(f0 ** 2 * L ** 2 / (gp * H2), symbol=Symbol("α'_2"), units='')

# Orography (non-dimensional)

hh = np.zeros(len(atmospheric_basis))
hh[1] = 0.2
h = ParameterField('h', u'h', hh, atmospheric_basis, inner_products_definition)

# Defining the fields
#######################
p1 = u'ψ_1'
psi1 = Field("psi1", p1, atmospheric_basis, inner_products_definition, units="", latex=r'\psi_1')
p2 = u'ψ_2'
psi2 = Field("psi2", p2, atmospheric_basis, inner_products_definition, units="", latex=r'\psi_2')


# --------------------------------
#
#   Top layer equation
#
# --------------------------------

# defining the LHS as the time derivative of the potential vorticity
psi1_vorticity = OperatorTerm(psi1, Laplacian, atmospheric_basis.coordinate_system)
dpsi12 = LinearTerm(psi2, prefactor=alphap_1)
dpsi11 = LinearTerm(psi1, prefactor=dalpha_1)
psi1_equation = Equation(psi1, lhs_terms=[psi1_vorticity, dpsi12, dpsi11])

# Defining the advection term
advection_term = vorticity_advection(psi1, psi1, atmospheric_basis.coordinate_system, sign=-1)
psi1_equation.add_rhs_terms(advection_term)

psi2_advec = Jacobian(psi1, psi2, atmospheric_basis.coordinate_system, sign=-1, prefactors=(alphap_1, alphap_1))
psi1_equation.add_rhs_terms(psi2_advec)

# adding the beta term
beta_term = OperatorTerm(psi1, D, x, prefactor=beta_nondim, sign=-1)
psi1_equation.add_rhs_term(beta_term)

# --------------------------------
#
#   Bottom layer equation
#
# --------------------------------

# defining the LHS as the time derivative of the potential vorticity
psi2_vorticity = OperatorTerm(psi2, Laplacian, atmospheric_basis.coordinate_system)
dpsi22 = LinearTerm(psi2, prefactor=alphap_2, sign=-1)
dpsi21 = LinearTerm(psi1, prefactor=alphap_2)
psi2_equation = Equation(psi2, lhs_terms=[psi2_vorticity, dpsi22, dpsi21])

# Defining the advection terms
advection_term = vorticity_advection(psi2, psi2, atmospheric_basis.coordinate_system, sign=-1)
psi2_equation.add_rhs_terms(advection_term)

psi1_advec = Jacobian(psi2, psi1, atmospheric_basis.coordinate_system, sign=-1, prefactors=(alphap_2, alphap_2))
psi2_equation.add_rhs_terms(psi1_advec)

# adding the beta term
beta_term = OperatorTerm(psi2, D, x, prefactor=beta_nondim, sign=-1)
psi2_equation.add_rhs_term(beta_term)

# adding an orographic term
orographic_term = Jacobian(psi2, h, atmospheric_basis.coordinate_system, sign=-1)

psi2_equation.add_rhs_terms(orographic_term)

# --------------------------------
#
#   Constructing the layer
#
# --------------------------------

layer1 = Layer()
layer1.add_equation(psi1_equation)

layer2 = Layer()
layer2.add_equation(psi2_equation)


# --------------------------------
#
#   Constructing the cake
#
# --------------------------------

cake = Cake()
cake.add_layer(layer1)
cake.add_layer(layer2)

# --------------------------------
#
#   Computing the tendencies
#
# --------------------------------

# computing the tensor
cake.compute_tensor(True, True)

# computing the tendencies
f, Df = cake.compute_tendencies()

# integrating
ic = np.random.rand(cake.ndim) * 0.1
res = solve_ivp(f, (0., 20000.), ic, method='DOP853')

ic = res.y[:, -1]
res = solve_ivp(f, (0., 20000.), ic, method='DOP853')

# plotting
plt.plot(res.y.T)
plt.show()
