
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos

import sys
import os
if os.path.basename(os.getcwd()) == 'LayerCake':
    sys.path.extend([os.path.abspath('./')])
else:
    sys.path.extend([os.path.abspath('../..')])

# importing all that is needed to create the cake
from layercake import *

# importing specific modules to create the model basis of functions
from layercake.basis import SphericalHarmonicsBasis
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition

##############################################################################################
#
# This script defines the Euler equation over a rotating sphere of radius one
# in numerical mode and then integrates (run) it, and plots the resulting trajectory.
#
# The model is coming from
#
# Constantin, A., & Germain, P. (2022). Stratospheric planetary flows from the perspective
# of the Euler equation on a rotating sphere. Archive for Rational Mechanics and Analysis, 245(1), 587-644.
# https://doi.org/10.1007/s00205-022-01791-3
#
# and is a very simplified model for the stratosphere.
#
##############################################################################################

# Defining parameter for the sphere
#####################################
omega = Parameter(9., symbol=symbols('ω'), latex=r'\omega', units='[s^-1]')
R = Parameter(1., symbol=symbols('R'))
parameters = [R]
# For an earth-like flow, omega is set to 9 (82 for Jupiter, 63 for Saturn, 13 for Neptune, 18 for Uranus).
# See reference above for more details.

# Defining basis of functions (modes) and inner products
#########################################################
basis = SphericalHarmonicsBasis(parameters, {'M': 4})
s = StandardSymbolicInnerProductDefinition(coordinate_system=basis.coordinate_system)

# coordinates
llambda = basis.coordinate_system.coordinates_symbol_as_list[0]
phi = basis.coordinate_system.coordinates_symbol_as_list[1]

# Defining the inverse of the cosine of the latitude
#########################################################
cos_inv = Expression(1 / (R**2 * cos(phi)), expression_parameters=(R,), latex=r'\frac{1}{R^2 \cos \phi}')

# Defining the field
p = u'ψ'
psi = Field("streamfunction", p, basis, s, units="[m^2][s^-2]", latex=r'\psi')

# Barotropic field equation definition
#######################################

# defining the LHS as the time derivative of the vorticity
vorticity = OperatorTerm(psi, Laplacian, basis.coordinate_system)
planetary_equation = Equation(psi, lhs_term=vorticity)

# defining the advection term
advection_term = vorticity_advection(psi, psi, basis.coordinate_system, sign=-1, prefactors=(cos_inv, cos_inv))

planetary_equation.add_rhs_terms(advection_term)


# defining the earth rotation f-term
a = Parameter(2. * float(omega), symbol=symbols('a'))
fterm = FunctionField(u'f', u'f', a * sin(phi), basis, expression_parameters=(a,),
                      inner_product_definition=s, latex=r'f')

rotation_advection_terms = Jacobian(psi, fterm, basis.coordinate_system, sign=-1, prefactors=(cos_inv, cos_inv))

planetary_equation.add_rhs_terms(rotation_advection_terms)


# Constructing the layer
#########################
layer = Layer()
layer.add_equation(planetary_equation)

# Constructing the cake
#########################
cake = Cake()
cake.add_layer(layer)

# Computing the tendencies and integrating
###########################################

# computing the tensor
cake.compute_tensor(True, True)

# computing the tendencies
f, Df = cake.compute_tendencies()

# integrating
ic = np.random.rand(cake.ndim) * 0.1
res = solve_ivp(f, (0., 100000.), ic, method='DOP853')
# Remark: The DOP853 is a 7th-order integrator which is not able to conserve the energy
# and dissipate here to a periodic orbit representing a rotating polar vortex.

# plotting
plt.plot(res.y.T)
plt.show()

