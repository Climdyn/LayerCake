import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import symbols, pi, sqrt, cos, sin

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
from layercake.variables.systems import PlanarCartesianCoordinateSystem
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition


##############################################################################################
#
# This script defines a barotropic one layer model over a beta plane
# in numerical mode and then integrates (run) it, and plots the resulting trajectory.
#
##############################################################################################

# Defining the domain
b = symbols('b')
b_param = Parameter(0.5, symbol=b)
parameters = [b]

cs = PlanarCartesianCoordinateSystem(extent=((0, 2*pi), (0, b*pi)))


# defining the basis
basis = SymbolicBasis(cs, parameters)

# coordinates
x = basis.coordinate_system.coordinates_symbol_as_list[0]
y = basis.coordinate_system.coordinates_symbol_as_list[1]

basis.functions.append(sqrt(2) * cos(y / b))
basis.functions.append(2 * cos(x) * sin(y / b))
basis.functions.append(2 * sin(x) * sin(y / b))
basis.functions.append(sqrt(2) * cos(2 * y / b))
basis.functions.append(2 * cos(x) * sin(2 * y / b))
basis.functions.append(2 * sin(x) * sin(2 * y / b))

basis.substitutions.append((b, float(b_param)))

inner_products = StandardSymbolicInnerProductDefinition(coordinate_system=basis.coordinate_system, optimizer='trig', kwargs={'conds': 'none'})

# Defining the field
p = u'ψ'
psi = Field("psi", p, basis, inner_products, units="[m^2][inner_products^-2]", latex=r'\psi')

# Defining the equation and LHS
# Laplacian
vorticity = OperatorTerm(psi, Laplacian, basis.coordinate_system)
barotropic_equation = Equation(psi, lhs_term=vorticity)

# Defining the advection term
advection_term = vorticity_advection(psi, psi, basis.coordinate_system, sign=-1)

barotropic_equation.add_rhs_terms(advection_term)

# adding an orographic term
gamma = Parameter(0.2, symbol=symbols(u'γ'), latex=r'\gamma')
hh = np.zeros(len(basis))
hh[1] = 1.05
h = ParameterField('h', u'h', hh, basis, inner_products)

orographic_term = Jacobian(psi, h, basis.coordinate_system, sign=-1, prefactors=(gamma, gamma))

barotropic_equation.add_rhs_terms(orographic_term)

# adding the beta term
beta = Parameter(1.25, symbol=symbols(u'β'), latex=r'\beta')
beta_term = OperatorTerm(psi, D, x, prefactor=beta, sign=-1)

barotropic_equation.add_rhs_term(beta_term)

# adding a Newtonian cooling
C_param = Parameter(0.1, symbol=symbols('C'))
newtonian_cooling1 = OperatorTerm(psi, Laplacian, basis.coordinate_system, prefactor=C_param, sign=-1)

psi_ast_array = np.zeros(len(basis))
r = -0.771
psi_ast_array[0] = 0.95
psi_ast_array[3] = r * psi_ast_array[0]
psi_ast = ParameterField('psi_ast', p+u'*', psi_ast_array, basis, inner_products, latex=r'\psi^\ast')
LinearTerm(psi_ast, inner_products, prefactor=C_param)
newtonian_cooling2 = OperatorTerm(psi_ast, Laplacian, basis.coordinate_system, prefactor=C_param)

barotropic_equation.add_rhs_terms((newtonian_cooling1, newtonian_cooling2))

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
res = solve_ivp(f, (0., 1000.), res.y[:, -1])

# plotting
plt.figure()
plt.plot(res.y.T)
plt.figure()
plt.plot(res.y[0], res.y[3])
plt.show()
