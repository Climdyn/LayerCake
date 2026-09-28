
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
from layercake.basis import SphericalHarmonicsBasis
from layercake.inner_products.definition import StandardSymbolicInnerProductDefinition


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

# Earth rotation angular speed
omega = Parameter(7.292e-5, symbol=Symbol(u'ω'), units='[s^-1]')

# Timescale parameter
T = Parameter(1. / (2. * float(omega)), symbol=symbols('T'), units='[s]')

# Defining the fields
#######################
p = u'ψ'
psi = Field("psi", p, basis, inner_products_definition, units="[m^2][^-2]", latex=r'\psi')

# Barotropic field equation definition
#######################################

# defining the LHS as the time derivative of the vorticity
vorticity = OperatorTerm(psi, Laplacian, basis.coordinate_system)
barotropic_equation = Equation(psi, lhs_terms=vorticity)

# defining the advection term
advection_term = vorticity_advection(psi, psi, basis.coordinate_system, sign=-1)
barotropic_equation.add_rhs_terms(advection_term)

# adding an orographic term (sin theta * h)
sin_th_h = np.array([-0.001711890606304873, 0.0022332703338084825,
                     0.001014260540193253, 0.002428461633157703,
                     0.0016877961736783316, 0.0015782594096663554,
                     -0.003619193452466132, -0.006561605391195204,
                     0.0009603132306877926, -0.00197822560376025,
                     0.019943023135503446, -0.00420363112509415,
                     0.0074384102492688975, -0.014195299050184684,
                     -1.137858057111412e-05, -0.001941233058648731,
                     0.0009283207552447782, 0.0012240310990215871,
                     -0.004083320319716952, -0.007690520373983598,
                     -0.005570890927646672, -0.0014451628606521009,
                     -0.0009667440544924891, 0.002455772979125063])
oro = ParameterField('h', u'h', sin_th_h, basis, inner_products_definition)

orographic_term = Jacobian(psi, oro, basis.coordinate_system, sign=-1)

barotropic_equation.add_rhs_terms(orographic_term)

# adding the beta term
beta_term = OperatorTerm(psi, D, llambda, sign=-1)
barotropic_equation.add_rhs_term(beta_term)

# adding a Newtonian cooling
C_param = Parameter(0.00294, symbol=symbols('C'))  # corresponds roughly to a 27 days timescale
newtonian_cooling1 = OperatorTerm(psi, Laplacian, basis.coordinate_system, prefactor=C_param, sign=-1)

# roughly realistic forcing at 500hPa
psi_ast_array = np.array([0.00537415,  0.00100627,  0.00158805, -0.00437707, -0.00563329,
                          -0.00749909, -0.01298291, -0.02065268, -0.00633784, -0.00088284,
                          0.01938229,  0.00521208,  0.01456594,  0.00077128, -0.00166069,
                          -0.00638115,  0.00681739,  0.00526438, -0.01511177, -0.01481776,
                          -0.01816274, -0.01217594, -0.0121209,  0.00287957])


psi_ast = ParameterField('psi_ast', p+u'*', psi_ast_array, basis, inner_products_definition, latex=r'\psi^\ast')
newtonian_cooling2 = LinearTerm(psi_ast, inner_products_definition, prefactor=C_param)
# newtonian_cooling2 = OperatorTerm(psi_ast, Laplacian, basis.coordinate_system, prefactor=C_param)

barotropic_equation.add_rhs_terms((newtonian_cooling1, newtonian_cooling2))

# Constructing the layer
#########################
layer = Layer()
layer.add_equation(barotropic_equation)

# Constructing the cake
#########################
cake = Cake()
cake.add_layer(layer)


# computing the tensor
######################
cake.compute_tensor(True, True
                    )

# computing the tendencies
##########################
f, Df = cake.compute_tendencies()


# integrating
#############
ic = np.random.rand(cake.ndim) * 0.1
res = solve_ivp(f, (0., 20000.), ic, method='DOP853')


# plotting
###########

# time in days
time = res.t * T / (24 * 3600)

# plotting the full time span
plt.figure()
plt.plot(time, res.y.T)

# plotting the last 100 time steps
# plotting
plt.plot(time[-100:], res.y.T[-100:])
plt.show()
