
"""

    Jacobian and vorticity module
    =============================

    This module defines a function to compute the terms for the Jacobian of fields
    in partial differential equations.
    A function to compute derived forms of the Jacobian like the vorticity advection is
    provided.

"""
from layercake.arithmetic.terms.operations import ProductOfTerms
from layercake.arithmetic.terms.operators import OperatorTerm, ComposedOperatorsTerm
from layercake.arithmetic.symbolic.operators import Laplacian, D
from layercake.arithmetic.symbolic.expressions import Expression
from layercake.variables.utils import combine_units, power_units


def Jacobian(field1, field2, coordinate_system, sign=1, prefactors=(None, None)):
    """Function returning a list of :class:`~layercake.arithmetic.terms.operations.ProductOfTerms` components of
    the partial differential equation's Jacobian:

    .. math:: J(\\psi, \\phi) = e_1 \\, e_2 \\, \\left(\\partial_{u_1} \\psi \\, \\partial_{u_2} \\phi - \\partial_{u_1} \\phi \\, \\partial_{u_2} \\psi \\right)

    where :math:`\\phi` and :math:`\\psi` are two fields defined on the model's domain,
    :math:`u_1, u_2` are the coordinates of the model, and :math:`e_1, e_2` are the inverse of the infinitesimal
    length of the coordinates.

    Parameters
    ----------
    field1: ~field.Field or ~field.ParameterField
        First field on which the Jacobian act (corresponds to the field :math:`\\psi` in the formula above).
    field2: ~field.Field or ~field.ParameterField
        Second field on which the Jacobian act (corresponds to the field :math:`\\phi` in the formula above).
    coordinate_system: ~systems.CoordinateSystem
        Coordinate system on which the model is defined.
    sign: int, optional
        Sign in front of the Jacobian term. Either +1 or -1.
        Default to +1.
    prefactors: tuple(~parameter.Parameter or ~expressions.Expression), optional
        2-tuple providing the prefactors in front of each of the two terms composing the Jacobian.
        This is added on top of the infinitesimal length prefactor.
        Must be specified as model parameters or symbolic expressions.

    Returns
    -------
    tuple(~operations.ProductOfTerms)
        2-tuple containing arithmetic terms representing each term of the Jacobian.
    """

    uc = coordinate_system.coordinates[0]
    vc = coordinate_system.coordinates[1]

    u = uc.symbol
    v = vc.symbol

    if uc.infinitesimal_length == 1 and vc.infinitesimal_length == 1:
        prefactor1 = prefactors[0]
        prefactor2 = prefactors[1]
    else:
        prefs = 1 / (uc.infinitesimal_length * vc.infinitesimal_length)
        if prefactors[0] is not None:
            prefs1 = prefactors[0].symbol * prefs
            prefactor1 = Expression(prefs1,
                                    expression_parameters=coordinate_system.parameters,
                                    units=combine_units(prefactors[0].units, combine_units(uc.units, vc.units, '+'), '-'),
                                    latex=prefs1._repr_latex_()[15:-1])
        else:
            prefactor1 = Expression(prefs,
                                    expression_parameters=coordinate_system.parameters,
                                    units=power_units(combine_units(uc.units, vc.units, '+'), -1),
                                    latex=prefs._repr_latex_()[15:-1])

        if prefactors[1] is not None:
            prefs2 = prefactors[2].symbol * prefs
            prefactor2 = Expression(prefs2,
                                    expression_parameters=coordinate_system.parameters,
                                    units=combine_units(prefactors[0].units, combine_units(uc.units, vc.units, '+'), '-'),
                                    latex=prefs2._repr_latex_()[15:-1])
        else:
            prefactor2 = Expression(prefs,
                                    expression_parameters=coordinate_system.parameters,
                                    units=power_units(combine_units(uc.units, vc.units, '+'), -1),
                                    latex=prefs._repr_latex_()[15:-1])

    du_field1 = OperatorTerm(field1, D, u, prefactor=prefactor1)
    du_field2 = OperatorTerm(field2, D, u)

    dv_field1 = OperatorTerm(field1, D, v, prefactor=prefactor2)
    dv_field2 = OperatorTerm(field2, D, v)

    jacobian1 = ProductOfTerms(du_field1, dv_field2, sign=sign)
    jacobian2 = ProductOfTerms(dv_field1, du_field2, sign=-sign)

    return jacobian1, jacobian2


def vorticity_advection(field1, field2, coordinate_system, sign=1, prefactors=(None, None)):
    """Function returning a list of :class:`~layercake.arithmetic.terms.operations.ProductOfTerms` components of
    the vorticity advection in the partial differential equation,
    provided by the expression :math:`J(\\psi, \\nabla^2 \\phi)`,
    where :math:`J` is the partial differential equation's Jacobian :func:`~layercake.arithmetic.terms.jacobian.Jacobian`:

    .. math:: J(\\psi, \\nabla^2 \\phi) = e_1 \\, e_2 \\, \\left(\\partial_{u_1} \\psi \\, \\partial_{u_2} \\nabla^2 \\phi - \\partial_{u_1} \\nabla^2 \\phi \\, \\partial_{u_2} \\psi \\right)

    where :math:`\\phi` and :math:`\\psi` are two fields defined on the model's domain,
    :math:`u_1, u_2` are the coordinates of the model, and :math:`e_1, e_2` are the inverse of the infinitesimal
    length of the coordinates.

    Parameters
    ----------
    field1: ~field.Field or ~field.ParameterField
        First field, advected by the vorticity.
    field2: ~field.Field or ~field.ParameterField
        Second field with which the vorticity :math:`\\nabla^2` is computed.
    coordinate_system: ~systems.CoordinateSystem
        Coordinate system on which the model is defined.
    sign: int, optional
        Sign in front of the vorticity advection term. Either +1 or -1.
        Default to +1.
    prefactors: tuple(~parameter.Parameter or ~expressions.Expression), optional
        2-tuple providing the prefactors in front of each of the two terms composing the Jacobian.
        This is added on top of the infinitesimal length prefactor.
        Must be specified as model parameters or symbolic expressions.

    Returns
    -------
    tuple(~operations.ProductOfTerms)
        2-tuple containing arithmetic terms representing each term of the vorticity advection.
    """

    uc = coordinate_system.coordinates[0]
    vc = coordinate_system.coordinates[1]

    u = uc.symbol
    v = vc.symbol

    if uc.infinitesimal_length == 1 and vc.infinitesimal_length == 1:
        prefactor1 = prefactors[0]
        prefactor2 = prefactors[1]
    else:
        prefs = 1 / (uc.infinitesimal_length * vc.infinitesimal_length)
        if prefactors[0] is not None:
            prefs1 = prefactors[0].symbol * prefs
            prefactor1 = Expression(prefs1,
                                    expression_parameters=coordinate_system.parameters,
                                    units=combine_units(prefactors[0].units, combine_units(uc.units, vc.units, '+'), '-'),
                                    latex=prefs1._repr_latex_()[15:-1])
        else:
            prefactor1 = Expression(prefs,
                                    expression_parameters=coordinate_system.parameters,
                                    units=power_units(combine_units(uc.units, vc.units, '+'), -1),
                                    latex=prefs._repr_latex_()[15:-1])

        if prefactors[1] is not None:
            prefs2 = prefactors[2].symbol * prefs
            prefactor2 = Expression(prefs2,
                                    expression_parameters=coordinate_system.parameters,
                                    units=combine_units(prefactors[0].units, combine_units(uc.units, vc.units, '+'), '-'),
                                    latex=prefs2._repr_latex_()[15:-1])
        else:
            prefactor2 = Expression(prefs,
                                    expression_parameters=coordinate_system.parameters,
                                    units=power_units(combine_units(uc.units, vc.units, '+'), -1),
                                    latex=prefs._repr_latex_()[15:-1])

    du_field1 = OperatorTerm(field1, D, u, prefactor=prefactor1)
    dv_field1 = OperatorTerm(field1, D, v, prefactor=prefactor2)

    du_lap_field2 = ComposedOperatorsTerm(field2, (D, Laplacian), (u, coordinate_system))

    dv_lap_field2 = ComposedOperatorsTerm(field2, (D, Laplacian), (v, coordinate_system))

    jacobian1 = ProductOfTerms(du_field1, dv_lap_field2, sign=sign)
    jacobian2 = ProductOfTerms(dv_field1, du_lap_field2, sign=-sign)

    return jacobian1, jacobian2
