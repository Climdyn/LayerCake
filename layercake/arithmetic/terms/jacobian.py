
"""

    Jacobian and vorticity module
    =============================

    This module defines a function to compute the terms for the Jacobian of fields
    in partial differential equations.
    A function to compute a usage of the Jacobian like the vorticity advection is
    provided.

"""
from layercake.arithmetic.terms.operations import ProductOfTerms
from layercake.arithmetic.terms.operators import OperatorTerm, ComposedOperatorsTerm
from layercake.arithmetic.symbolic.operators import Laplacian, D


def Jacobian(field1, field2, coordinate_system, sign=1, prefactors=(None, None)):
    """Function returning a list of :class:`ProductOfTerms` components of
    the partial differential equation's Jacobian:

    .. math:

        J(\\psi, \\phi) = \\partial_{u_1} \\psi \\partial_{u_2} \\phi - \\partial_{u_1} \\phi \\partial_{u_2} \\psi

    where :math:`\\phi` and :math:`\\psi` are two fields defined on the model's domain,
    and :math:`u_1, u_2` are the coordinates of the model.

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
    prefactors: tuple(~parameter.Parameter), optional
        2-tuple providing the prefactors in front of each of the two terms composing the Jacobian.
        Must be specified as model parameters.

    Returns
    -------
    tuple(~operations.ProductOfTerms)
        2-tuple containing arithmetic terms representing each term of the Jacobian.
    """

    u = coordinate_system.coordinates_symbol_as_list[0]
    v = coordinate_system.coordinates_symbol_as_list[1]

    du_field1 = OperatorTerm(field1, D, u, prefactor=prefactors[0])
    du_field2 = OperatorTerm(field2, D, u, prefactor=prefactors[1])

    dv_field1 = OperatorTerm(field1, D, v)
    dv_field2 = OperatorTerm(field2, D, v)

    jacobian1 = ProductOfTerms(du_field1, dv_field2, sign=sign)
    jacobian2 = ProductOfTerms(dv_field1, du_field2, sign=-sign)

    return jacobian1, jacobian2


def vorticity_advection(field1, field2, coordinate_system, sign=1, prefactors=(None, None)):
    """Function returning a list of :class:`ProductOfTerms` components of
    the vorticity advection in the partial differential equation,
    provided by the expression :math:`J(\\psi, \\nabla^2 \\phi)`,
    where :math:`J` is the partial differential equation's Jacobian :func:`~jacobian.Jacobian`:

    .. math:

        J(\\psi, \\nabla^2 \\phi) = \\partial_{u_1} \\psi \\partial_{u_2} \\nabla^2 \\phi - \\partial_{u_1} \\nabla^2 \\phi \\partial_{u_2} \\psi

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
    prefactors: tuple(~parameter.Parameter), optional
        2-tuple providing the prefactors in front of each of the two terms of the Jacobian of the
        vorticity advection. Must be specified as model parameters.

    Returns
    -------
    tuple(~operations.ProductOfTerms)
        2-tuple containing arithmetic terms representing each term of the vorticity advection.
    """

    u = coordinate_system.coordinates_symbol_as_list[0]
    v = coordinate_system.coordinates_symbol_as_list[1]

    du_field1 = OperatorTerm(field1, D, u, prefactor=prefactors[0])
    dv_field1 = OperatorTerm(field1, D, v, prefactor=prefactors[1])

    du_lap_field2 = ComposedOperatorsTerm(field2, (D, Laplacian), (u, coordinate_system))

    dv_lap_field2 = ComposedOperatorsTerm(field2, (D, Laplacian), (v, coordinate_system))

    jacobian1 = ProductOfTerms(du_field1, dv_lap_field2, sign=sign)
    jacobian2 = ProductOfTerms(dv_field1, du_lap_field2, sign=-sign)

    return jacobian1, jacobian2
