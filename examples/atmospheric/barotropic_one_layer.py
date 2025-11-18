import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import symbols

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
# This script defines a barotropic one layer model over a beta plane
# in numerical mode and then integrates (run) it, and plots the resulting trajectory.
#
##############################################################################################

# Defining the domain
_n = symbols('n')
n = Parameter(1.3, symbol=_n)
parameters = [n]
b = contiguous_channel_basis(1, 2, parameters)
s = StandardSymbolicInnerProductDefinition(coordinate_system=b.coordinate_system)

# coordinates
x = b.coordinate_system.coordinates_symbol_as_list[0]
y = b.coordinate_system.coordinates_symbol_as_list[1]

# Defining the field
p = u'ψ'
psi = Field("psi", p, b, s, units="[m^2][s^-2]", latex=r'\psi')

# Defining the equation and LHS
# Laplacian
vorticity = OperatorTerm(psi, Laplacian, b.coordinate_system)
barotropic_equation = Equation(psi, lhs_term=vorticity)

# Defining the advection term
advection_term = vorticity_advection(psi, psi, b.coordinate_system, sign=-1)

barotropic_equation.add_rhs_terms(advection_term)

# adding an orographic term
g = 0.1
gamma = symbols(u'γ')
gammap = Parameter(g, symbol=gamma, latex=r'\gamma')
hh = np.zeros(len(b))
hh[1] = 1.
h = ParameterField('h', u'h', hh, b, s)

orographic_term = Jacobian(psi, h, b.coordinate_system, sign=-1, prefactors=(gammap, gammap))

barotropic_equation.add_rhs_terms(orographic_term)

# adding the beta term
betaa = symbols(u'β')
beta = Parameter(0.20964969238375256, symbol=betaa, latex=r'\beta')
betaterm = OperatorTerm(psi, D, x, prefactor=beta, sign=-1)

barotropic_equation.add_rhs_term(betaterm)

# adding a friction
kdd = symbols('k_d')
kd = Parameter(0.05, symbol=kdd)
friction = OperatorTerm(psi, Laplacian, b.coordinate_system, prefactor=kd, sign=-1)
barotropic_equation.add_rhs_term(friction)

# adding an interaction with a background streamfunction
Cdd = symbols('C')
Cd = Parameter(0.05, symbol=Cdd)
rr = np.zeros(len(b))
rr[0] = 0.3
C = ParameterField('eta', u'η', rr, b, s, latex=r'\eta')
CT = OperatorTerm(C, Laplacian, b.coordinate_system, prefactor=Cd)

barotropic_equation.add_rhs_term(CT)

# constructing the layer
layer = Layer()
layer.add_equation(barotropic_equation)

# constructing the cake
cake = Cake()
cake.add_layer(layer)

# computing the tensor
cake.compute_tensor(True, True, compute_inner_products_kwargs={'timeout': True}
                    )
# computing the tendencies
f, Df = cake.compute_tendencies()

# integrating
ic = np.random.rand(cake.ndim) * 0.1
res = solve_ivp(f, (0., 1000.), ic)

# plotting
plt.plot(res.y.T)
plt.show()
