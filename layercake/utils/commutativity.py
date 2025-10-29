
"""

    Commutativity utility module
    ============================

    Defines useful functions to deal with enabling or disabling commutativity in |Sympy| expressions.

"""

from sympy import Symbol


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
