
import sys
import os

import unittest
import numpy as np
from sympy import symbols, Symbol

from qgs.params.params import QgParams
from qgs.inner_products import analytic
from qgs.tensors.qgtensor import QgsTensor

path = os.path.abspath('./')
base = os.path.basename(path)
if base == 'test':
    sys.path.extend([os.path.abspath('../')])
else:
    sys.path.extend([path])

from layercake.basis.planar_fourier import contiguous_channel_basis, contiguous_basin_basis
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

from test.test_base import TestQgsBase


real_eps = 20 * np.finfo(np.float64).eps


class State:

    qgs_tensor = None
    layercake_cake = None


class TestMaooamQgsNumerical(TestQgsBase):

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
        # Setting MAOOAM default parameters:
        params = QgParams()
        params.set_atmospheric_channel_fourier_modes(2, 2)
        params.set_oceanic_basin_fourier_modes(2, 4)

        # Setting MAOOAM default parameters
        params.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5, 'r': 1.e-7,
                           'h': 136.5, 'd': 1.1e-7})
        params.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3,
                                               'hlambda': 15.06})
        params.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})

        params.atemperature_params.set_insolation(103.3333, 0)
        params.gotemperature_params.set_insolation(310, 0)

        aip = analytic.AtmosphericAnalyticInnerProducts(params)  # , stored=False)
        oip = analytic.OceanicAnalyticInnerProducts(params)  # , stored=False)
        aip.connect_to_ocean(oip)

        self.state.qgs_tensor = QgsTensor(params=params, atmospheric_inner_products=aip, oceanic_inner_products=oip)

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

        # Computing qgs version of the MAOOAM tensor:
        # Setting MAOOAM default parameters:
        # Characteristic length scale (L_y / pi)
        L = Parameter(1591549.4309189534, symbol=Symbol('L'), units='[m]')

        # Domain aspect ratio
        n = Parameter(1.5, symbol=symbols('n'))

        # Coriolis parameter at the middle of the domain
        f0 = Parameter(1.032e-4, symbol=Symbol('f_0'), units='[s^-1]')

        # Pressure difference between the two atmospheric layers
        deltap = Parameter(5.e4, symbol=Symbol('Δp'), units='[Pa]')

        # Static stability of the atmosphere
        sigma = Parameter(2.1581898457499433e-6, symbol=Symbol('σ'), units='[m^2][s^-2][Pa^-2]')

        # Meridional gradient of the Coriolis parameter at phi_0
        beta = Parameter(1.620094191522763e-11, symbol=Symbol(u'β'), units='[m^-1][s^-1]')

        # atmosphere bottom friction coefficient
        kd = Parameter(2.9928e-6, symbol=Symbol('k_d'), units='[s^-1]')

        # Atmosphere internal friction coefficient
        kdp = Parameter(2.9928e-6, symbol=Symbol("k_d'"), units='[s^-1]')

        # Friction between the ocean and the atmosphere
        d = Parameter(1.1e-7, symbol=Symbol('d'), units='[s^-1]')

        # Friction of the ocean with the bottom
        r = Parameter(1.e-7, symbol=Symbol('r'), units='[s^-1]')

        # Physical constants
        # gas constant of dry air
        rr = Parameter(287.058, symbol=Symbol('R'), units='[J][kg^-1][K^-1]')
        # Stefan-Boltzmann constant
        sb = Parameter(5.67e-08, symbol=Symbol('σ_B'), units='[J][m^-2][s^-1][K^-4]')

        # Sensible + turbulent heat exchange between ocean/ground and atmosphere
        hlambda = Parameter(15.06, symbol=Symbol('λ'), units='[J][m^-2][K^-1][s^-1]')

        # Stationary solution for the 0-th order atmospheric temperature
        T0a = Parameter(289.3, symbol=Symbol('T_{a, 0}'), units='[K]')

        # Stationary solution for the 0-th order atmospheric temperature
        T0o = Parameter(301.46, symbol=Symbol('T_{o, 0}'), units='[K]')

        # Specific heat capacity of the ocean
        gamma_o = Parameter(560000000.0, symbol=Symbol('γ_o'), units='[J][m^-2][K^-1]')

        # Specific heat capacity of the atmosphere
        gamma_a = Parameter(10000000.0, symbol=Symbol('γ_a'), units='[J][m^-2][K^-1]')

        # Depth of the water layer of the ocean
        h = Parameter(136.5, symbol=Symbol('h'), units='[m]')

        # Emissivity coefficient for the grey-body atmosphere
        eps = Parameter(0.7, symbol=Symbol('ε'))

        # Reduced gravity
        gp = Parameter(0.031, symbol=Symbol("g'"), units='[m][s^-2]')

        # Defining the domain
        ######################

        parameters = [n]
        atmospheric_basis = contiguous_channel_basis(2, 2, parameters)
        oceanic_basis = contiguous_basin_basis(2, 4, parameters)
        s = StandardSymbolicInnerProductDefinition(coordinate_system=atmospheric_basis.coordinate_system)
        # coordinates
        x = atmospheric_basis.coordinate_system.coordinates_symbol_as_list[0]
        y = atmospheric_basis.coordinate_system.coordinates_symbol_as_list[1]

        # Derived (non-dimensional) parameters
        #######################################

        sigma_nondim = Parameter((sigma * deltap ** 2) / (L ** 2 * f0 ** 2), symbol=sigma.symbol, units='')
        beta_nondim = Parameter(beta * L / f0, symbol=beta.symbol, units='')
        d_nondim = Parameter(d / f0, symbol=d.symbol)
        r_nondim = Parameter(r / f0, symbol=r.symbol)
        a = Parameter(2 / sigma_nondim, symbol=Symbol('a'))
        kd_deriv = Parameter(0.5 * kd / f0, symbol=kd.symbol)
        kdp_deriv = Parameter(2 * kdp / f0, symbol=kdp.symbol)
        LR = Parameter((gp * h) ** 0.5 / f0, symbol=Symbol('L_R'), units='[m]')
        G = Parameter(- L ** 2 / LR ** 2, symbol=Symbol('G'))
        Lpo = hlambda / (gamma_o * f0)
        Lpa = hlambda / (gamma_a * f0)
        sbpo = 4 * sb * T0o ** 3 / (gamma_o * f0)
        sbpa = 8 * eps * sb * T0a ** 3 / (gamma_o * f0)
        LSBpo = 2 * eps * sb * T0o ** 3 / (gamma_a * f0)
        LSBpa = 8 * eps * sb * T0a ** 3 / (gamma_a * f0)

        # Defining the insolation
        # In the ocean
        Cov = np.zeros(len(atmospheric_basis))
        Cov[0] = 310
        Cpo = ParameterField("Co'", Symbol("Co'"), Cov / (gamma_o * f0) * rr / (f0 ** 2 * L ** 2), atmospheric_basis, s)

        # In the atmosphere
        Cav = np.zeros(len(atmospheric_basis))
        Cav[0] = 310. / 3.
        Cpa = ParameterField("Ca'", Symbol("Ca'"), a * Cav / (gamma_a * f0) * rr / (f0 ** 2 * L ** 2) / 2, atmospheric_basis, s)

        # Defining the fields
        #######################
        p = u'ψ_a'
        psi_a = Field("psi_a", p, atmospheric_basis, s, units="[m^2][s^-2]", latex=r'\psi_{\rm a}')
        tt = u'θ_a'
        theta_a = Field("theta_a", tt, atmospheric_basis, s, units="[m^2][s^-2]", latex=r'\theta_{\rm a}')

        p_o = u'ψ_o'
        psi_o = Field("psi_o", p_o, oceanic_basis, s, units="[m^2][s^-2]", latex=r'\psi_{\rm o}')
        tt_o = u'δT_o'
        deltaT_o = Field("deltaT_o", tt_o, oceanic_basis, s, units="[m^2][s^-2]", latex=r'\delta T_{\rm o}')

        # --------------------------------------------------------
        #
        #   Barotropic atmospheric field equation
        #
        # --------------------------------------------------------

        # Defining the equation and LHS
        # Laplacian
        vorticity = OperatorTerm(psi_a, Laplacian, atmospheric_basis.coordinate_system)
        barotropic_equation = Equation(psi_a, lhs_terms=vorticity)

        # Defining the advection term
        advection_term1 = vorticity_advection(psi_a, psi_a, atmospheric_basis.coordinate_system, sign=-1)
        advection_term2 = vorticity_advection(theta_a, theta_a, atmospheric_basis.coordinate_system, sign=-1)

        barotropic_equation.add_rhs_terms(advection_term1)
        barotropic_equation.add_rhs_terms(advection_term2)

        # adding the beta term
        beta_term = OperatorTerm(psi_a, D, x, prefactor=beta_nondim, sign=-1)
        barotropic_equation.add_rhs_term(beta_term)

        # adding the atmospheric friction
        friction = OperatorTerm(psi_a, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv, sign=-1)
        barotropic_equation.add_rhs_term(friction)

        ofriction = OperatorTerm(theta_a, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv)
        barotropic_equation.add_rhs_term(ofriction)

        ocfriction = OperatorTerm(psi_o, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv)
        barotropic_equation.add_rhs_term(ocfriction)

        # --------------------------------------------------------
        #
        #   Baroclinic atmospheric field equation
        #
        # --------------------------------------------------------

        # Defining the equation and LHS
        # Laplacian
        vorticity = OperatorTerm(theta_a, Laplacian, atmospheric_basis.coordinate_system)

        lin_lhs = LinearTerm(theta_a, prefactor=a, sign=-1)
        lhs = AdditionOfTerms(lin_lhs, vorticity)
        baroclinic_equation = Equation(theta_a, lhs_terms=lhs)

        # Defining the advection terms
        advection_term1 = vorticity_advection(psi_a, theta_a, atmospheric_basis.coordinate_system, sign=-1)
        advection_term2 = vorticity_advection(theta_a, psi_a, atmospheric_basis.coordinate_system, sign=-1)

        baroclinic_equation.add_rhs_terms(advection_term1)
        baroclinic_equation.add_rhs_terms(advection_term2)

        # adding the beta term
        beta_term = OperatorTerm(theta_a, D, x, prefactor=beta_nondim, sign=-1)
        baroclinic_equation.add_rhs_term(beta_term)

        # adding the friction with the ocean
        friction = OperatorTerm(psi_a, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv, sign=1)
        baroclinic_equation.add_rhs_term(friction)

        ofriction = OperatorTerm(theta_a, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv, sign=-1)
        baroclinic_equation.add_rhs_term(ofriction)

        ocfriction = OperatorTerm(psi_o, Laplacian, atmospheric_basis.coordinate_system, prefactor=kd_deriv, sign=-1)
        baroclinic_equation.add_rhs_term(ocfriction)

        # adding the atmospheric friction
        ground_friction = OperatorTerm(theta_a, Laplacian, atmospheric_basis.coordinate_system, prefactor=kdp_deriv, sign=-1)
        baroclinic_equation.add_rhs_term(ground_friction)

        # adding jacobian from thermal wind relation
        thermal = Jacobian(psi_a, theta_a, atmospheric_basis.coordinate_system, prefactors=(a, a))
        baroclinic_equation.add_rhs_terms(thermal)

        # adding heat exchange scheme

        la1 = Symbol('λ1')
        la1p = Parameter(a * (Lpa + LSBpa), symbol=la1)

        la1t = LinearTerm(theta_a, prefactor=la1p, sign=1)

        la2 = Symbol('λ2')
        la2p = Parameter(a * (0.5 * Lpa + LSBpo), symbol=la2)

        la2t = LinearTerm(deltaT_o, prefactor=la2p, sign=-1)

        baroclinic_equation.add_rhs_terms((la1t, la2t))

        # adding the insolation
        Cat = LinearTerm(Cpa, sign=-1)
        baroclinic_equation.add_rhs_term(Cat)

        # --------------------------------------------------------
        #
        #   Shallow-water oceanic field equation
        #
        # --------------------------------------------------------

        # Defining the equation and LHS
        # Laplacian
        vorticity = OperatorTerm(psi_o, Laplacian, oceanic_basis.coordinate_system)
        lin_lhs = LinearTerm(psi_o, prefactor=G)
        lhs = AdditionOfTerms(lin_lhs, vorticity)
        oceanic_equation = Equation(psi_o, lhs_terms=lhs)

        # Defining the advection term
        advection_term = vorticity_advection(psi_o, psi_o, oceanic_basis.coordinate_system, sign=-1)
        oceanic_equation.add_rhs_terms(advection_term)

        # adding the beta term
        beta_term = OperatorTerm(psi_o, D, x, prefactor=beta_nondim, sign=-1)
        oceanic_equation.add_rhs_term(beta_term)

        # adding friction with the ocean
        friction = OperatorTerm(psi_a, Laplacian, oceanic_basis.coordinate_system, prefactor=d_nondim, sign=1)
        oceanic_equation.add_rhs_term(friction)

        ofriction = OperatorTerm(theta_a, Laplacian, oceanic_basis.coordinate_system, prefactor=d_nondim, sign=-1)
        oceanic_equation.add_rhs_term(ofriction)

        ocfriction = OperatorTerm(psi_o, Laplacian, oceanic_basis.coordinate_system, prefactor=d_nondim, sign=-1)
        oceanic_equation.add_rhs_term(ocfriction)

        # adding friction at the bottom of the ocean
        bfriction = OperatorTerm(psi_o, Laplacian, oceanic_basis.coordinate_system, prefactor=r_nondim, sign=-1)
        oceanic_equation.add_rhs_term(bfriction)

        # --------------------------------------------------------
        #
        #   Oceanic temperature field equation
        #
        # --------------------------------------------------------

        # Defining the equation and LHS
        # Laplacian
        lhs = LinearTerm(deltaT_o)
        ocean_temperature_equation = Equation(deltaT_o, lhs_terms=lhs)

        # Defining the advection term
        advection_term = Jacobian(psi_o, deltaT_o, oceanic_basis.coordinate_system, sign=-1)
        ocean_temperature_equation.add_rhs_terms(advection_term)

        # adding heat exchange scheme
        lo3 = Symbol('λ3')
        lo3p = Parameter(Lpo + sbpo, symbol=lo3)

        lo3t = LinearTerm(deltaT_o, prefactor=lo3p, sign=-1)

        lo4 = Symbol('λ4')
        lo4p = Parameter(2 * Lpo + sbpa, symbol=lo4)

        lo4t = LinearTerm(theta_a, prefactor=lo4p, sign=1)

        ocean_temperature_equation.add_rhs_terms((lo3t, lo4t))

        # adding the insolation
        Cot = LinearTerm(Cpo, sign=1)
        ocean_temperature_equation.add_rhs_term(Cot)

        # --------------------------------
        #
        #   Constructing the layers
        #
        # --------------------------------

        atmospheric_layer = Layer(name='Atmosphere')
        atmospheric_layer.add_equation(barotropic_equation)
        atmospheric_layer.add_equation(baroclinic_equation)

        oceanic_layer = Layer(name='Ocean')
        oceanic_layer.add_equation(oceanic_equation)
        oceanic_layer.add_equation(ocean_temperature_equation)

        # --------------------------------
        #
        #   Constructing the cake
        #
        # --------------------------------

        cake = Cake()
        cake.add_layer(atmospheric_layer)
        cake.add_layer(oceanic_layer)

        # --------------------------------
        #
        #   Computing the tensor
        #
        # --------------------------------

        cake.compute_tensor(True, True, compute_inner_products_kwargs={'timeout': True}
                            )

        self.state.layercake_cake = cake
        tensor = cake.tensor

        for coo, val in zip(tensor.coords.T, tensor.data):
            _string_format(tfunc, 'tensor', coo, val)

    def layercake_jacobian_outputs(self, output_func=None):

        if output_func is None:
            self.layercake_values.clear()
            tfunc = self.save_layercake
        else:
            tfunc = output_func

        tensor = self.state.layercake_cake.jacobian_tensor

        for coo, val in zip(tensor.coords.T, tensor.data):
            _string_format(tfunc, 'tensor', coo, val)


def _string_format(func, symbol, indices, value):
    if abs(value) >= real_eps:
        s = symbol
        for i in indices:
            s += "[" + str(i) + "]"
        s += " = % .5E" % value
        func(s)


if __name__ == "__main__":
    unittest.main()
