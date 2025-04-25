
from sympy.core.decorators import call_highest_priority
from sympy import Expr, Matrix, Mul, Add, diff, zeros
from sympy.core.numbers import Zero

from layercake.utils.commutativity import enable_commutativity
from layercake.variables.systems import CoordinateSystem

# courtesy of https://stackoverflow.com/questions/15463412/differential-operator-usable-in-matrix-form-in-python-module-sympy


class D(Expr):
    _op_priority = 11.
    is_commutative = False

    def __init__(self, *variables, **assumptions):
        super(D, self).__init__()
        self.evaluate = False
        self.variables = variables

    def __repr__(self):
        return 'D%s' % str(self.variables)

    def __str__(self):
        return self.__repr__()

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return Mul(other, self)

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        if isinstance(other, D):
            variables = self.variables + other.variables
            return D(*variables)
        if isinstance(other, Matrix):
            other_copy = other.copy()
            for i, elem in enumerate(other):
                other_copy[i] = self * elem
            return other_copy

        if self.evaluate:
            return diff(other, *self.variables)
        else:
            return Mul(self, other)

    def __pow__(self, other):
        variables = self.variables
        for i in range(other-1):
            variables += self.variables
        return D(*variables)


def _diff(expr, *variables):
    if isinstance(expr, D):
        expr.variables += variables
        return D(*expr.variables)
    if isinstance(expr, Matrix):
        expr_copy = expr.copy()
        for i, elem in enumerate(expr):
            expr_copy[i] = diff(elem, *variables)
        return expr_copy
    return diff(expr, *variables)


def _evaluate_mul(expr):
    end = 0
    if expr.args:
        if isinstance(expr.args[-1], D):
            if len(expr.args[:-1]) == 1:
                cte = expr.args[0]
                return Zero()
            end = -1
    for i in range(len(expr.args)-1+end, -1, -1):
        arg = expr.args[i]
        if isinstance(arg, Add):
            arg = _evaluate_add(arg)
        if isinstance(arg, Mul):
            arg = _evaluate_mul(arg)
        if isinstance(arg, D):
            left = Mul(*expr.args[:i])
            right = Mul(*expr.args[i+1:])
            right = _diff(right, *arg.variables)
            ans = left * right
            return _evaluate_mul(ans)
    return expr


def _evaluate_add(expr):
    newargs = []
    for arg in expr.args:
        if isinstance(arg, Mul):
            arg = _evaluate_mul(arg)
        if isinstance(arg, Add):
            arg = _evaluate_add(arg)
        if isinstance(arg, D):
            arg = Zero()
        newargs.append(arg)
    return Add(*newargs)


def evaluate_expr(expr):
    if isinstance(expr, Matrix):
        for i, elem in enumerate(expr):
            elem = elem.expand()
            expr[i] = evaluate_expr(elem)
        return enable_commutativity(expr)
    expr = expr.expand()
    if isinstance(expr, Mul):
        expr = _evaluate_mul(expr)
    elif isinstance(expr, Add):
        expr = _evaluate_add(expr)
    elif isinstance(expr, D):
        expr = Zero()
    return expr


def Nabla(coordinates_system):
    if not isinstance(coordinates_system, CoordinateSystem):
        raise ValueError('Nabla only take coordinates systems as input.')

    derivative_list = list()
    for coord in coordinates_system.coordinates:
        derivative_list.append(Mul(coord.infinitesimal_length**(-1), D(coord.symbol), evaluate=False))
    return Matrix([derivative_list])


def Divergence(coordinates_system):

    if not isinstance(coordinates_system, CoordinateSystem):
        raise ValueError('Divergence only take coordinates systems as input.')

    derivative_list = list()
    volume = coordinates_system.infinitesimal_volume
    for coord in coordinates_system.coordinates:
        derivative_list.append(Mul(volume**(-1), Mul(D(coord.symbol), volume, evaluate=False), evaluate=False))

    return Matrix([derivative_list])


def Laplacian(coordinates_system):
    if not isinstance(coordinates_system, CoordinateSystem):
        raise ValueError('Laplacian only take coordinates systems as input.')
    nabla = Nabla(coordinates_system)
    divergence = Divergence(coordinates_system)
    laplacian = zeros(*nabla.shape)
    for i in range(len(nabla)):
        laplacian[i] = Mul(divergence[i] * nabla[i], evaluate=False)
    return laplacian

