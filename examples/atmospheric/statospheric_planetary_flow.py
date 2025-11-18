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
from layercake.basis import SphericalHarmonicsBasis
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition

# Defining parameter for the sphere

_omega = symbols('ω')
omega = Parameter(7.292e-5, symbol=_omega, latex=r'\omega', units='[s^-1]')
_R = symbols('R')
R = Parameter(1., symbol=_R)
parameters = [R]

# defining basis of functions (modes) and inner products
basis = SphericalHarmonicsBasis(parameters, {'M': 4})
s = StandardSymbolicInnerProductDefinition(coordinate_system=basis.coordinate_system)

# coordinates
llambda = basis.coordinate_system.coordinates_symbol_as_list[0]
phi = basis.coordinate_system.coordinates_symbol_as_list[1]

# Defining the field
p = u'ψ'
psi = Field("streamfunction", p, basis, s, units="[m^2][s^-2]", latex=r'\psi')


# Defining the equation and LHS
# Laplacian
vorticity = OperatorTerm(psi, Laplacian, basis.coordinate_system)
planetary_equation = Equation(psi, lhs_term=vorticity)

# Defining the advection term
advection_term = vorticity_advection(psi, psi, basis.coordinate_system, sign=-1)

planetary_equation.add_rhs_terms(advection_term)


# Defining the earth rotation f-term
sin_func = basis.find_functions(1, 0)
rr = np.zeros(len(basis))
rr[sin_func] = np.sqrt(np.pi / 3)
sin_theta = ParameterField('2 sin phi', u'sin ϕ', rr, basis, s, latex=r'2 \sin \phi')

rotation_advection_terms = Jacobian(psi, sin_theta, basis.coordinate_system, sign=-1, prefactors=(omega, omega))

planetary_equation.add_rhs_terms(rotation_advection_terms)


# constructing the layer
layer = Layer()
layer.add_equation(planetary_equation)

# constructing the cake
cake = Cake()
cake.add_layer(layer)

# # computing the tensor
# cake.compute_tensor(True, True
#                     )
# # computing the tendencies
# f, Df = cake.compute_tendencies()
#
# # integrating
# ic = np.random.rand(cake.ndim) * 0.1
# res = solve_ivp(f, (0., 1000.), ic)
#
# # plotting
# plt.plot(res.y.T)
# plt.show()
#
