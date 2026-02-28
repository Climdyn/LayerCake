
import sys
import os

import unittest
import numpy as np
from sympy import symbols, Symbol, N

from qgs.params.params import QgParams
from qgs.inner_products import analytic
from qgs.tensors.qgtensor import QgsTensor


path = os.path.abspath('./')
base = os.path.basename(path)
if base == 'test':
    sys.path.extend([os.path.abspath('../')])
else:
    sys.path.extend([path])

from layercake.basis.planar_fourier import contiguous_channel_basis
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
from layercake.utils.symbolic_tensor import get_coords_and_values_from_tensor

from test.test_base import TestQgsBase


real_eps = 10 * np.finfo(np.float64).eps


class State:

    qgs_tensor = None
    layercake_cake = None


class TestRP1982QgsSymbolic(TestQgsBase):

    def test_against_qgs(self):
        self.__class__.state = State()
        self.check_lists(self.qgs_outputs, self.layercake_outputs)

    def test_jacobian_against_qgs(self):
        self.check_lists(self.qgs_jacobian_outputs, self.layercake_jacobian_outputs)

    def qgs_outputs(self, output_func=None):

        if output_func is None:
            self.qgs_values.clear()
            tfunc = self.save_qgs
        else:
            tfunc = output_func

        # Computing qgs version of the RP1982 tensor:
        # Setting RP1982 default parameters:
        # Model parameters instantiation with some non-default specs
        params = QgParams({'phi0_npi': np.deg2rad(50.) / np.pi, 'hd': 0.045})
        # Mode truncation at the wavenumber 2 in both x and y spatial coordinate
        params.set_atmospheric_channel_fourier_modes(2, 2)
        # Setting the orography depth and the meridional temperature gradient
        params.ground_params.set_orography(0.2, 1)
        params.atemperature_params.set_thetas(0.1, 0)

        aip = analytic.AtmosphericAnalyticInnerProducts(params)

        self.state.qgs_tensor = QgsTensor(params=params, atmospheric_inner_products=aip)

        for coo, val in zip(self.state.qgs_tensor.tensor.coords.T, self.state.qgs_tensor.tensor.data):
            _string_format(tfunc, 'tensor', coo, val)

    def qgs_jacobian_outputs(self, output_func=None):

        if output_func is None:
            self.qgs_values.clear()
            tfunc = self.save_qgs
        else:
            tfunc = output_func

        for coo, val in zip(self.state.qgs_tensor.jacobian_tensor.coords.T, self.state.qgs_tensor.jacobian_tensor.data):
            _string_format(tfunc, 'tensor', coo, val)

    def layercake_outputs(self, output_func=None):

        if output_func is None:
            self.layercake_values.clear()
            tfunc = self.save_layercake
        else:
            tfunc = output_func

        # Computing qgs version of the RP1982 tensor:
        # Setting RP1982 default parameters:
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

        # atmosphere bottom friction coefficient
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
        s = StandardSymbolicInnerProductDefinition(coordinate_system=atmospheric_basis.coordinate_system, optimizer='trig', kwargs={'conds': 'none'})
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
        h = ParameterField('h', u'h', hh, atmospheric_basis, s)

        # Equilibrium temperature
        rr = np.zeros(len(atmospheric_basis))
        rr[0] = 0.1
        Tf = ParameterField('T', u'T', rr, atmospheric_basis, s)

        # Defining the fields
        #######################
        p = u'ψ'
        psi = Field("psi", p, atmospheric_basis, s, units="[m^2][s^-2]", latex=r'\psi')
        tt = u'θ'
        theta = Field("theta", tt, atmospheric_basis, s, units="[m^2][s^-2]", latex=r'\theta')

        # --------------------------------
        #
        #   Barotropic field equation
        #
        # --------------------------------

        # Defining the equation and LHS
        # Laplacian
        vorticity = OperatorTerm(psi, Laplacian, atmospheric_basis.coordinate_system)
        barotropic_equation = Equation(psi, lhs_terms=vorticity)

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

        # Defining the equation and LHS
        # Laplacian
        vorticity = OperatorTerm(theta, Laplacian, atmospheric_basis.coordinate_system)

        lin_lhs = LinearTerm(theta, prefactor=a, sign=-1)
        lhs = AdditionOfTerms(lin_lhs, vorticity)
        baroclinic_equation = Equation(theta, lhs_terms=lhs)

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
        #   Computing the tensor
        #
        # --------------------------------

        cake.compute_tensor(False,
                            True,
                            basis_subs=True,
                            parameters_subs=cake.parameters
                            )

        self.state.layercake_cake = cake
        tensor = cake.tensor

        coords_val = get_coords_and_values_from_tensor(tensor, 'tuple')
        for coo_val in coords_val:
            coo = coo_val[:-1]
            val = coo_val[-1]
            _string_format(tfunc, 'tensor', coo, N(val))

    def layercake_jacobian_outputs(self, output_func=None):

        if output_func is None:
            self.layercake_values.clear()
            tfunc = self.save_layercake
        else:
            tfunc = output_func

        tensor = self.state.layercake_cake.jacobian_tensor

        coords_val = get_coords_and_values_from_tensor(tensor, 'tuple')
        for coo_val in coords_val:
            coo = coo_val[:-1]
            val = coo_val[-1]
            _string_format(tfunc, 'tensor', coo, N(val))


def _string_format(func, symbol, indices, value):
    if abs(value) >= real_eps:
        s = symbol
        for i in indices:
            s += "[" + str(i) + "]"
        s += " = % .5E" % value
        func(s)


if __name__ == "__main__":
    unittest.main()
