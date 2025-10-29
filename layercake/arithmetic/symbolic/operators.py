
"""

    Operators definition module
    ===========================

    This module defines various symbolic operators (mainly differential ones)
    acting on the fields of the partial differential equations in `Sympy`_ expression.

    .. _Sympy: https://www.sympy.org/

"""


from sympy.core.decorators import call_highest_priority
from sympy import Expr, Matrix, Mul, Add, diff
from sympy.core.numbers import Zero

from layercake.utils.commutativity import enable_commutativity
from layercake.variables.systems import CoordinateSystem

# courtesy of https://stackoverflow.com/questions/15463412/differential-operator-usable-in-matrix-form-in-python-module-sympy


class D(Expr):
    """Symbolic differential operator acting on `Sympy`_ expression.
    Inspired by `this post <https://stackoverflow.com/questions/15463412/differential-operator-usable-in-matrix-form-in-python-module-sympy>`_.

    .. _Sympy: https://www.sympy.org/

    Parameters
    ----------
    *variables: ~sympy.core.symbol.Symbol
        Variables with respect to which the operator differentiate.
        The number of variables indicate the order of the derivative.

    Attributes
    ----------
    variables: list(~sympy.core.symbol.Symbol)
        Variables with respect to which the operator differentiate.
        The number of variables indicate the order of the derivative.
    evaluate: bool
        Whether the expression resulting from the action of the operator is
        evaluated.
        Default to `False`.
    latex: str
        LaTeX representation of the operator.

    """
    _op_priority = 11.
    is_commutative = False

    def __init__(self, *variables, **assumptions):
        super(D, self).__init__()
        self.evaluate = False
        self.variables = variables
        latexes = list()
        for var in variables:
            if hasattr(var, 'latex'):
                if var.latex is not None:
                    latexes.append(var.latex)
                    continue
            if hasattr(var, 'symbol'):
                if var.symbol is not None:
                    latexes.append(str(var.symbol))
                    continue
            latexes.append(str(var))

        if len(variables) > 1:
            self.latex = r'\frac{\partial^' + str(len(variables)) + r'}{'
        else:
            self.latex = r'\frac{\partial}{'

        for var in latexes[:-1]:
            self.latex += r'\partial ' + var + ' '
        self.latex += r'\partial ' + latexes[-1] + r'}'

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
    """Evaluate a given `Sympy`_ expression.

    .. _Sympy: https://www.sympy.org/

    Parameters
    ----------
    expr: ~sympy.core.expr.Expr
        The expression to evaluate.

    Returns
    -------
    ~sympy.core.expr.Expr
        The evaluated expression.

    """
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


def _latex_repr(r):
    def wrapper(f):
        f.latex = r
        return f
    return wrapper


@_latex_repr(r'\nabla')
def Nabla(coordinate_system):
    """Function returning the Nabla (Del - :math:`\\nabla`) operator associated with a given
    coordinate system.

    Notes
    -----
    The returned expression has an additional `latex` attribute.

    Parameters
    ----------
    coordinate_system: ~coordinates.CoordinateSystem
        Coordinate system for which the :math:`\\nabla` operator must be returned.

    Returns
    -------
    ~sympy.core.expr.Expr
        The :math:`\\nabla` operator associated with the coordinate system.

    """
    if not isinstance(coordinate_system, CoordinateSystem):
        raise ValueError('Nabla only take coordinates systems as input.')

    derivative_list = list()
    for coord in coordinate_system.coordinates:
        derivative_list.append(Mul(coord.infinitesimal_length**(-1), D(coord.symbol), evaluate=False))

    mat = Matrix([derivative_list])
    mat.latex = r'\nabla'
    return mat


@_latex_repr(r'\nabla \cdot')
def Divergence(coordinate_system):
    """Function returning the divergence (:math:`\\nabla \\cdot`) operator associated with a given
    coordinate system.

    Notes
    -----
    The returned expression has an additional `latex` attribute.

    Parameters
    ----------
    coordinate_system: ~coordinates.CoordinateSystem
        Coordinate system for which the divergence operator must be returned.

    Returns
    -------
    ~sympy.core.expr.Expr
        The divergence operator associated with the coordinate system.

    """

    if not isinstance(coordinate_system, CoordinateSystem):
        raise ValueError('Divergence only take coordinates systems as input.')

    derivative_list = list()
    volume = coordinate_system.infinitesimal_volume
    for coord in coordinate_system.coordinates:
        derivative_list.append(Mul(volume**(-1), Mul(D(coord.symbol), volume, evaluate=False), evaluate=False))

    mat = Matrix([derivative_list])
    mat.latex = r'\nabla \cdot'
    return mat


class _Add(Add):

    def __new__(cls, *args, **kwargs):
        try:
            latex = kwargs.pop('latex')
        except KeyError:
            latex = ''
        a = Add.__new__(cls, *args, **kwargs)
        a._latex = latex

        return a

    @property
    def latex(self):
        return self._latex


@_latex_repr(r'\nabla^2')
def Laplacian(coordinate_system):
    """Function returning the Laplacian (:math:`\\nabla^2`) operator associated with a given
    coordinate system.

    Notes
    -----
    The returned expression has an additional `latex` attribute.

    Parameters
    ----------
    coordinate_system: ~coordinates.CoordinateSystem
        Coordinate system for which the Laplacian operator must be returned.

    Returns
    -------
    ~sympy.core.expr.Expr
        The Laplacian operator associated with the coordinate system.

    """
    if not isinstance(coordinate_system, CoordinateSystem):
        raise ValueError('Laplacian only take coordinates systems as input.')
    nabla = Nabla(coordinate_system)
    divergence = Divergence(coordinate_system)
    laplacian = Mul(divergence[0] * nabla[0], evaluate=False)
    for i in range(1, len(nabla)):
        laplacian = _Add(laplacian, Mul(divergence[i] * nabla[i], evaluate=False), latex=r'\nabla^2')
    return laplacian
