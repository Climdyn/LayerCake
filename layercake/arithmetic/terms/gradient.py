
"""

    Gradient module
    ===============

    This module defines a function to compute the terms for the scalar product of gradients of fields
    in partial differential equations.
    A function to compute derived forms of the gradient products involving the vorticity is also
    provided.

"""
from layercake.arithmetic.terms.operations import ProductOfTerms
from layercake.arithmetic.terms.operators import OperatorTerm, ComposedOperatorsTerm
from layercake.arithmetic.symbolic.operators import Laplacian, D
from layercake.arithmetic.symbolic.expressions import Expression
from layercake.variables.utils import combine_units, power_units


def gradients_product(field1, field2, coordinate_system, sign=1, prefactors=(None, None)):
    """Function returning a list of :class:`~layercake.arithmetic.terms.operations.ProductOfTerms` components of
    the partial differential equation's scalar product of gradients:

    .. math:: P(\\psi, \\phi) = e_1^2 \\, \\partial_{u_1} \\psi \\, \\partial_{u_1} \\phi + e_2^2 \\, \\partial_{u_2} \\psi \\, \\partial_{u_2} \\phi

    where :math:`\\phi` and :math:`\\psi` are two fields defined on the model's domain,
    :math:`u_1, u_2` are the coordinates of the model, and :math:`e_1, e_2` are the inverse of the infinitesimal
    length of the coordinates (which can be functions of the :math:`u_1, u_2` coordinates).

    Parameters
    ----------
    field1: ~field.Field or ~field.ParameterField
        Field on which the first gradient of the scalar product act (corresponds to the field :math:`\\psi` in the formula above).
    field2: ~field.Field or ~field.ParameterField
        Field on which the second gradient of the scalar product act (corresponds to the field :math:`\\phi` in the formula above).
    coordinate_system: ~systems.CoordinateSystem
        Coordinate system on which the model is defined.
    sign: int, optional
        Sign in front of the term. Either +1 or -1.
        Default to +1.
    prefactors: tuple(~parameter.Parameter or ~expressions.Expression), optional
        2-tuple providing the prefactors in front of each of the two terms composing the scalar product of gradients.
        This is added on top of the infinitesimal length prefactor.
        Must be specified as model parameters or symbolic expressions.

    Returns
    -------
    tuple(~operations.ProductOfTerms)
        2-tuple containing arithmetic terms representing each term of the scalar product of gradients.
    """

    uc = coordinate_system.coordinates[0]
    vc = coordinate_system.coordinates[1]

    u = uc.symbol
    v = vc.symbol

    uunits = combine_units(uc.units, uc.infinitesimal_length_units, '+')
    vunits = combine_units(vc.units, vc.infinitesimal_length_units, '+')

    if uc.infinitesimal_length == 1 and vc.infinitesimal_length == 1:
        prefactor1 = prefactors[0]
        prefactor2 = prefactors[1]
    else:
        prefs = 1 / (uc.infinitesimal_length ** 2)
        if prefactors[0] is not None:
            prefs1 = prefactors[0].symbol * prefs
            prefactor1 = Expression(prefs1,
                                    expression_parameters=coordinate_system.parameters,
                                    units=combine_units(prefactors[0].units, power_units(uunits, 2), '-'),
                                    latex=prefs1._repr_latex_()[15:-1])
        else:
            prefactor1 = Expression(prefs,
                                    expression_parameters=coordinate_system.parameters,
                                    units=power_units(uunits, -2),
                                    latex=prefs._repr_latex_()[15:-1])

        prefs = 1 / (vc.infinitesimal_length ** 2)
        if prefactors[1] is not None:
            prefs2 = prefactors[1].symbol * prefs
            prefactor2 = Expression(prefs2,
                                    expression_parameters=coordinate_system.parameters,
                                    units=combine_units(prefactors[1].units, power_units(vunits, 2), '-'),
                                    latex=prefs2._repr_latex_()[15:-1])
        else:
            prefactor2 = Expression(prefs,
                                    expression_parameters=coordinate_system.parameters,
                                    units=power_units(vunits, -2),
                                    latex=prefs._repr_latex_()[15:-1])

    du_field1 = OperatorTerm(field1, D, u, prefactor=prefactor1)
    du_field2 = OperatorTerm(field2, D, u)

    dv_field1 = OperatorTerm(field1, D, v, prefactor=prefactor2)
    dv_field2 = OperatorTerm(field2, D, v)

    product1 = ProductOfTerms(du_field1, du_field2, sign=sign)
    product2 = ProductOfTerms(dv_field1, dv_field2, sign=sign)

    return product1, product2


def vorticity_gradients_product(field1, field2, coordinate_system, sign=1, prefactors=(None, None)):
    """Function returning a list of :class:`~layercake.arithmetic.terms.operations.ProductOfTerms` components of
    the vorticity advection in the partial differential equation,
    provided by the expression :math:`P(\\psi, \\nabla^2 \\phi)`,
    where :math:`P` is the partial differential equation's scalar product of gradients :func:`~layercake.arithmetic.terms.gradient.gradients_product`:

    .. math:: P(\\psi, \\nabla^2 \\phi) = e_1^2 \\, \\partial_{u_1} \\psi \\, \\partial_{u_1} \\nabla^2 \\phi + e_2^2 \\, \\partial_{u_2} \\psi \\, \\partial_{u_2} \\nabla^2 \\phi

    where :math:`\\phi` and :math:`\\psi` are two fields defined on the model's domain,
    :math:`u_1, u_2` are the coordinates of the model, and :math:`e_1, e_2` are the inverse of the infinitesimal
    length of the coordinates (which can be functions of the :math:`u_1, u_2` coordinates).

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

    uunits = combine_units(uc.units, uc.infinitesimal_length_units, '+')
    vunits = combine_units(vc.units, vc.infinitesimal_length_units, '+')

    if uc.infinitesimal_length == 1 and vc.infinitesimal_length == 1:
        prefactor1 = prefactors[0]
        prefactor2 = prefactors[1]
    else:
        prefs = 1 / (uc.infinitesimal_length ** 2)
        if prefactors[0] is not None:
            prefs1 = prefactors[0].symbol * prefs
            prefactor1 = Expression(prefs1,
                                    expression_parameters=coordinate_system.parameters,
                                    units=combine_units(prefactors[0].units, power_units(uunits, 2), '-'),
                                    latex=prefs1._repr_latex_()[15:-1])
        else:
            prefactor1 = Expression(prefs,
                                    expression_parameters=coordinate_system.parameters,
                                    units=power_units(uunits, -2),
                                    latex=prefs._repr_latex_()[15:-1])

        prefs = 1 / (vc.infinitesimal_length ** 2)
        if prefactors[1] is not None:
            prefs2 = prefactors[1].symbol * prefs
            prefactor2 = Expression(prefs2,
                                    expression_parameters=coordinate_system.parameters,
                                    units=combine_units(prefactors[1].units, power_units(vunits, 2), '-'),
                                    latex=prefs2._repr_latex_()[15:-1])
        else:
            prefactor2 = Expression(prefs,
                                    expression_parameters=coordinate_system.parameters,
                                    units=power_units(vunits, -2),
                                    latex=prefs._repr_latex_()[15:-1])

    du_field1 = OperatorTerm(field1, D, u, prefactor=prefactor1)
    dv_field1 = OperatorTerm(field1, D, v, prefactor=prefactor2)

    du_lap_field2 = ComposedOperatorsTerm(field2, (D, Laplacian), (u, coordinate_system))
    dv_lap_field2 = ComposedOperatorsTerm(field2, (D, Laplacian), (v, coordinate_system))

    product1 = ProductOfTerms(du_field1, du_lap_field2, sign=sign)
    product2 = ProductOfTerms(dv_field1, dv_lap_field2, sign=sign)

    return product1, product2
