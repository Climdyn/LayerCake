
"""

    Linear arithmetic term definition module
    ========================================

    This module defines linear terms in partial differential equations.
    The corresponding objects hold the symbolic representation of the terms and their decomposition
    on given function basis.

"""

from layercake.arithmetic.terms.base import SingleArithmeticTerm
from layercake.arithmetic.utils import sproduct


class LinearTerm(SingleArithmeticTerm):
    """Linear term in a partial differential equation, of the form :math:`a \\psi(u_1, u_2)`,
    where :math:`u_1, u_2` are the coordinates of the model, :math:`a` is a prefactor, and where :math:`\\psi` is
    the field solution of the equation.

    Parameters
    ----------
    field: ~field.Field
        The field over which the partial differential equation acts.
    inner_product_definition: InnerProductDefinition, optional
        Object defining the integral representation of the inner product that is used to compute the term representation on a given function basis.
        If not provided, it will use the inner product definition found in the `field` object.
        Default to using the inner product definition found in the `field` object.
    prefactor: parameter.Parameter, optional
        Prefactor in front of the single term.
        Must be specified as a model parameter.
    name: str, optional
        Name of the term(s).
    sign: int, optional
        Sign in front of the term(s). Either +1 or -1.
        Default to +1.

    Attributes
    ----------
    field: ~field.Field
        The field over which the partial differential equation acts.
    name: str
        Name of the term.
    sign: int
        Sign in front of the term. Either +1 or -1.
    inner_products: None or ~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.tensor.array.ImmutableSparseNDimArray or sparse.COO(float)
        The inner products tensor of the term.
        Set initially to `None` (not computed).
    inner_product_definition: InnerProductDefinition
        Object defining the integral representation of the inner product that is used to compute the term representation on a given function basis.
    prefactor: parameter.Parameter
        Prefactor in front of the single term.
    """

    def __init__(self, field, inner_product_definition=None, prefactor=None, name='', sign=1):

        SingleArithmeticTerm.__init__(self, field, inner_product_definition, prefactor, name, sign=sign)

    @property
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: The symbolic expression of the term(s). Only contains symbols."""
        if self.prefactor is None:
            return sproduct(self.sign, self.field.symbol)
        else:
            return sproduct(self.sign * self.prefactor.symbol, self.field.symbol)

    @property
    def numerical_expression(self):
        """~sympy.core.expr.Expr: The numeric expression of the term(s), with parameters replaced by their numerical value."""
        if self.prefactor is None:
            return sproduct(self.sign, self.field.symbol)
        else:
            return sproduct(self.sign * self.prefactor, self.field.symbol)

    @property
    def latex(self):
        """str: Return a LaTeX representation of the term(s)."""
        if self.sign > 0:
            s = f'+ '
        else:
            s = f'- '
        if self.prefactor is None:
            return s + self.field.latex
        if hasattr(self.prefactor, 'latex'):
            if self.prefactor.latex is not None:
                s += f'{self.prefactor.latex} '
                return s + self.field.latex
        if hasattr(self.prefactor, 'symbol'):
            if self.prefactor.symbol is not None:
                s += f'{self.prefactor.symbol} '
                return s + self.field.latex

        s += f'{self.prefactor} '
        return s + self.field.latex
