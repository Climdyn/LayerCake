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
from layercake.arithmetic.terms.operators import OperatorTerm, ComposedOperatorsTerm

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
R_E = Parameter(6371e3, symbol=Symbol('R_E'), units='[m]')

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
Rp = Parameter(1., symbol=Symbol("R"), units='')
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
R1d = Parameter(R_E ** 2 / R1 ** 2, symbol=Symbol("R'_1^-2"), units='', latex="R'_1^{-2}")
R2d = Parameter(R_E ** 2 / R2 ** 2, symbol=Symbol("R'_2^-2"), units='', latex="R'_2^{-2}")

# Rescaled timescales
tau_Rp = Parameter(tau_R / T, symbol=Symbol(u"τ'_R"), units='')
tau_Hp = Parameter(tau_H / T, symbol=Symbol(u"τ'_H"), units='')
tau_Ep = Parameter(tau_E / T, symbol=Symbol(u"τ'_E"), units='')

# Products
a_R1 = Parameter(R1d / tau_Rp, symbol=Symbol('a_R1'), units='', latex='a_{R1}')
a_H = Parameter((21 * 2) ** -4 / tau_Hp, symbol=Symbol("a_H"), units='')
a_H1 = Parameter((21 * 2) ** -4 * R1d / tau_Hp, symbol=Symbol("a_H1"), units='', latex='a_{H1}')

# Climatological forcings
s1r = np.zeros(len(basis))
s1r[0] = 0.1
S1p = ParameterField("S1p", "S'_1", s1r, basis, inner_products_definition, latex="S'_1")


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

# defining the advection term
psi1_advection_term = vorticity_advection(psi_1, psi_1, basis.coordinate_system, sign=-1)
psi1_equation.add_rhs_terms(psi1_advection_term)

# defining the Jacobian term
psi12_jacobian = Jacobian(psi_1, psi_2, basis.coordinate_system, prefactors=(R1d, R1d), sign=-1)
psi1_equation.add_rhs_terms(psi12_jacobian)

# adding the beta term
beta_term = OperatorTerm(psi_1, D, llambda, sign=-1)
psi1_equation.add_rhs_term(beta_term)

# radiative linear terms (could reuse LHS ones in a second pass)
lin_rad_rhs11 = LinearTerm(psi_1, prefactor=a_R1)
lin_rad_rhs12 = LinearTerm(psi_2, prefactor=a_R1, sign=-1)
psi1_equation.add_rhs_terms([lin_rad_rhs11, lin_rad_rhs12])

# horizontal PV diffusion terms
operators = (Laplacian,) * 10
operators_args = (basis.coordinate_system,) * 10
psi1_hdiffv = ComposedOperatorsTerm(psi_1, operators, operators_args, prefactor=a_H, sign=-1)
psi1_equation.add_rhs_term(psi1_hdiffv)

operators = (Laplacian,) * 8
operators_args = (basis.coordinate_system,) * 8
psi1_hdiff1 = ComposedOperatorsTerm(psi_1, operators, operators_args, prefactor=a_H1)
psi1_equation.add_rhs_term(psi1_hdiff1)

psi1_hdiff2 = ComposedOperatorsTerm(psi_2, operators, operators_args, prefactor=a_H1, sign=-1)
psi1_equation.add_rhs_term(psi1_hdiff2)

# climatological forcing
psi1_forcing = LinearTerm(S1p)
psi1_equation.add_rhs_term(psi1_forcing)


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
#
# # computing the tensor
# cake.compute_tensor(True, True)
#
# # computing the tendencies
# f, Df = cake.compute_tendencies()
#
# # integrating
# ic = np.random.rand(cake.ndim) * 0.1
# res = solve_ivp(f, (0., 10000000.), ic)
#
# ic = res.y[:, -1]
# res = solve_ivp(f, (0., 1000000.), ic)
#
# # plotting
# plt.plot(res.y[21], res.y[29])
# plt.show()
