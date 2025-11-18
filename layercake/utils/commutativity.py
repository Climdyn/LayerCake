
"""

    Commutativity utility module
    ============================

    Defines useful functions to deal with enabling or disabling commutativity in |Sympy| expressions.

"""

from sympy import Symbol, Add, Mul
from sympy.core.numbers import One


def enable_commutativity(expr):
    """Enable commutativity of a given Sympy expression.

    Parameters
    ----------
    expr: ~sympy.core.expr.Expr
        The expression on which to enable commutativity.

    Returns
    -------
    ~sympy.core.expr.Expr
        The expression with the commutativity enabled.
    """
    replacements = {s: Symbol(s.name) for s in expr.free_symbols}
    return expr.xreplace(replacements)


def disable_commutativity(expr):
    """Disable commutativity of a given Sympy expression.

    Parameters
    ----------
    expr: ~sympy.core.expr.Expr
        The expression on which to disable commutativity.

    Returns
    -------
    ~sympy.core.expr.Expr
        The expression with the commutativity disabled.
    """
    replacements = {s: Symbol(s.name, commutative=False) for s in expr.free_symbols}
    return expr.xreplace(replacements)


class NonCommutativeOne(One):
    """Dummy class for a non-commuting one (1)."""

    def __init__(self):
        One.__init__(self)
        self.is_commutative = False
        # self._op_priority = 11.

    @property
    def latex(self):
        return ""


def expand_and_deal_with_constant(expr):

    expanded = expr.expand()
    if isinstance(expanded, Add):
        new_args = list()
        for arg in expanded.args:
            if arg.is_constant():
                new_args.append(Mul(arg, NonCommutativeOne(), evaluate=False))
            else:
                new_args.append(arg)

        new_expr = new_args[0]
        for arg in new_args[1:]:
            new_expr = new_expr + arg
        return new_expr
    elif expr.is_constant():
        new_expr = Mul(expr, NonCommutativeOne(), evaluate=False)
        return new_expr

    else:
        return expr.expand()
