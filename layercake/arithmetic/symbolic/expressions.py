
"""
    Expressions definition module
    =============================

    This module defines mathematical expression to be inserted in the models equations.

"""


class Expression(object):
    """ Class defining a general mathematical expression in the equations.
    Can be used for example as prefactor for arithmetic terms.

    Parameters
    ----------
    symbolic_expression: ~sympy.core.expr.Expr
        A |Sympy| expression to represent the field mathematical expression in symbolic expressions.
    expression_parameters: None or list(~parameter.Parameter), optional
        List of parameters appearing in the symbolic expression.
        If `None`, assumes that no parameters are appearing there.
    units: str, optional
        The units of the variable.
        Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
        Empty by default.
    latex: str, optional
        Latex string representing the variable.
        Empty by default.

    Attributes
    ----------
    symbolic_expression: ~sympy.core.expr.Expr
        A |Sympy| expression to represent the field mathematical expression in symbolic expressions.
    units: str
        The units of the variable.
        Should be specified by joining atoms like `'[unit^power]'`, e.g '`[m^2][s^-2][Pa^-2]'`.
    latex: str, optional
        Latex string representing the variable.
    expression_parameters: None or list(~parameter.Parameter), optional
        List of parameters appearing in the symbolic expression.
        If `None`, assumes that no parameters are appearing there.

    """

    def __init__(self, symbolic_expression, expression_parameters=None, units="", latex=None):

        self.symbolic_expression = symbolic_expression
        self.expression_parameters = expression_parameters
        self.units = units
        self.latex = latex

    @property
    def symbol(self):
        """~sympy.core.expr.Expr: Synonym for the symbolic expression, to be accepted as prefactor."""
        return self.symbolic_expression
    @property
    def numerical_expression(self):
        """~sympy.core.expr.Expr: The numeric expression, i.e. with parameters replaced by their numerical value."""
        substitutions = list()
        if self.expression_parameters is None:
            expr_parameters = list()
        else:
            expr_parameters = self.expression_parameters
        for param in expr_parameters:
            substitutions.append((param.symbol, float(param)))
        return self.symbolic_expression.subs(substitutions)

    def __str__(self):
        return self.symbolic_expression

    def __repr__(self):
        return self.__str__()

