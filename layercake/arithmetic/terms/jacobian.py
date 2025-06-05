
from layercake.arithmetic.terms.operations import ProductOfTerms
from layercake.arithmetic.terms.operators import OperatorTerm, ComposedOperatorsTerm
from layercake.arithmetic.symbolic.operators import Laplacian, D


def Jacobian(field1, field2, coordinate_system, sign=1, prefactors=(None, None)):

    u = coordinate_system.coordinates_symbol_as_list[0]
    v = coordinate_system.coordinates_symbol_as_list[1]

    du_field1 = OperatorTerm(field1, D, u, prefactor=prefactors[0])
    du_field2 = OperatorTerm(field2, D, u, prefactor=prefactors[1])

    dv_field1 = OperatorTerm(field1, D, v)
    dv_field2 = OperatorTerm(field2, D, v)

    jacobian1 = ProductOfTerms(du_field1, dv_field2, sign=sign)
    jacobian2 = ProductOfTerms(dv_field1, du_field2, sign=-sign)

    return jacobian1, jacobian2


def advection(field1, field2, coordinate_system, sign=1, prefactors=(None, None)):

    u = coordinate_system.coordinates_symbol_as_list[0]
    v = coordinate_system.coordinates_symbol_as_list[1]

    du_field1 = OperatorTerm(field1, D, u, prefactor=prefactors[0])
    dv_field1 = OperatorTerm(field1, D, v, prefactor=prefactors[1])

    du_lap_field2 = ComposedOperatorsTerm(field2, (D, Laplacian), (u, coordinate_system))

    dv_lap_field2 = ComposedOperatorsTerm(field2, (D, Laplacian), (v, coordinate_system))

    jacobian1 = ProductOfTerms(du_field1, dv_lap_field2, sign=sign)
    jacobian2 = ProductOfTerms(dv_field1, du_lap_field2, sign=-sign)

    return jacobian1, jacobian2
