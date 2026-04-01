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
from layercake.basis.spherical_harmonics import SphericalHarmonicsBasis
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition


##############################################################################################
#
# This script defines the Marshall and Molteni model
# (https://journals.ametsoc.org/view/journals/atsc/50/12/1520-0469_1993_050_1792_taduop_2_0_co_2.xml)
# in numerical mode and then integrates (run) it, and plots the resulting trajectory in 2D.
#
##############################################################################################

# Setting some parameters
##########################

# Average radius of Earth
R = Parameter(6371e3, symbol=Symbol('R'), units='[m]')

# Rossby deformation radius for the 200-500-hPa layer
R1 = Parameter(700.e3, symbol=Symbol('R_1'), units='[m]')

# Rossby deformation radius for the 500-800-hPa layer
R2 = Parameter(450.e3, symbol=Symbol('R_2'), units='[m]')

# Scale height
H0 = Parameter(9e3, symbol=Symbol('H_0'), units='[m]')

# Earth rotation angular speed
omega = Parameter(7.292e-5, symbol=Symbol(u'ω'), units='[s^-1]')

# Radiative timescales
tau_R = Parameter(25 * 24 * 3600, symbol=Symbol(u'τ_R'), units='[s]')

# Horizontal diffusion timescale
tau_H = Parameter(2 * 24 * 3600, symbol=Symbol(u'τ_H'), units='[s]')

# Ekman dissipation timescale
tau_E = Parameter(3 * 24 * 3600, symbol=Symbol(u'τ_E'), units='[s]')


# Defining the domain
######################

# Adimensional Earth sphere radius parameter
Rp = Parameter(1., symbol=Symbol("R'"), units='')
parameters = [Rp]

basis = SphericalHarmonicsBasis(parameters, {'M': 4})
inner_products_definition = StandardSymbolicInnerProductDefinition(coordinate_system=basis.coordinate_system,
                                                                   optimizer='trig', kwargs={'conds': 'none'})
# coordinates
llambda = basis.coordinate_system.coordinates_symbol_as_list[0]
phi = basis.coordinate_system.coordinates_symbol_as_list[1]

# Derived dimensional parameters
################################

# Timescale parameter
T = Parameter(1. / (2. * float(omega)), symbol=symbols('T'), units='[s]')

# Derived (non-dimensional) parameters
#######################################

# Inverse of the square of the rescaled Rossby radius
R1d = Parameter(R**2 / R1**2, symbol=Symbol("R'_1^-2"), units='')
R2d = Parameter(R**2 / R2**2, symbol=Symbol("R'_2^-2"), units='')

# Rescaled timescales
tau_Rp = Parameter(tau_R / T, symbol=Symbol(u'τ_R'), units='[s]')
tau_Hp = Parameter(tau_H / T, symbol=Symbol(u'τ_H'), units='[s]')
tau_Ep = Parameter(tau_E / T, symbol=Symbol(u'τ_E'), units='[s]')


# Defining the fields
#######################

# 200 hPa streamfunction
p1 = u'ψ_1'
psi_1 = Field("psi_1", p1, basis, inner_products_definition, units="[m^2][s^-2]", latex=r'\psi_1')

# 500 hPa streamfunction
p2 = u'ψ_2'
psi_2 = Field("psi_2", p2, basis, inner_products_definition, units="[m^2][s^-2]", latex=r'\psi_2')

# 800 hPa streamfunction
p3 = u'ψ_3'
psi_3 = Field("psi_3", p3, basis, inner_products_definition, units="[m^2][s^-2]", latex=r'\psi_3')


# --------------------------------------------------------
#
#   First layer equation (200 hPa)
#
# --------------------------------------------------------

# defining LHS as the time derivative of the vorticity
psi1_vorticity = OperatorTerm(psi_1, Laplacian, basis.coordinate_system)
lin_lhs11 = LinearTerm(psi_1, prefactor=R1d, sign=-1)
lin_lhs12 = LinearTerm(psi_2, prefactor=R1d)
psi1_equation = Equation(psi_1, lhs_terms=[psi1_vorticity, lin_lhs12, lin_lhs11])

# Defining the advection term
psi1_advection_term = vorticity_advection(psi_1, psi_1, basis.coordinate_system, sign=-1)
psi1_equation.add_rhs_terms(psi1_advection_term)

# Defining the Jacobian term
psi12_jacobian = Jacobian(psi_1, psi_2, basis.coordinate_system, prefactors=(R1d, R1d), sign=-1)
psi1_equation.add_rhs_terms(psi12_jacobian)

# adding the beta term
beta_term = OperatorTerm(psi_1, D, llambda, sign=-1)
psi1_equation.add_rhs_term(beta_term)

# linear terms
a12 = Parameter()
lin_rhs11 = LinearTerm(psi_1, prefactor=R1d, sign=-1)
lin_rhs12 = LinearTerm(psi_2, prefactor=R1d)


# --------------------------------
#
#   Constructing the layers
#
# --------------------------------

layer1 = Layer(name='Top layer (200 hPa)')
layer1.add_equation(psi1_equation)

# --------------------------------
#
#   Constructing the cake
#
# --------------------------------

cake = Cake()
cake.add_layer(layer1)


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
res = solve_ivp(f, (0., 10000000.), ic)

ic = res.y[:, -1]
res = solve_ivp(f, (0., 1000000.), ic)

# plotting
plt.plot(res.y[21], res.y[29])
plt.show()
