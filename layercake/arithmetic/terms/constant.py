
"""

    Constant arithmetic term definition module
    ==========================================

    This module defines a constant terms in partial differential equations,
    i.e. a time-invariant spatial pattern.
    The corresponding objects hold the symbolic representation of the terms and their decomposition
    on given function basis.

"""

from layercake.arithmetic.terms.base import ArithmeticTerms
from layercake.variables.field import ParameterField

from layercake.utils.commutativity import disable_commutativity
from layercake.arithmetic.utils import sproduct
from sympy import Lambda


class ConstantTerm(ArithmeticTerms):
    """Constant term in a partial differential equation, i.e. a time-invariant field :math:`C(u_1, u_2)`
    where :math:`u_1, u_2` are the coordinates of the model.
    Time-invariant here means :math:`\\partial_t C = 0`.
    Allows to directly introduce constant terms into the ODEs.

    Parameters
    ----------
    parameter_field: ~field.ParameterField
        The field provided as a parameter field object which contains
        also the Galerkin expansion coefficients of the field on a given basis.
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
    """
    def __init__(self, parameter_field, name='', sign=1):

        if not isinstance(parameter_field, ParameterField):
            raise ValueError('Input field for constant field must be a ParameterField object.')

        ArithmeticTerms.__init__(self, name, sign=sign)

        self._rank = 1
        self.field = parameter_field

    @property
    def terms(self):
        return [self]

    @property
    def _symbolic_expressions_list(self):
        return [self.symbolic_expression]

    @property
    def _numerical_expressions_list(self):
        return [self.numerical_expression]

    @property
    def _symbolic_functions_list(self):
        return [self.symbolic_function]

    @property
    def _numerical_functions_list(self):
        return [self.numerical_function]

    @property
    def symbolic_function(self):
        """~sympy.core.expr.Expr: The symbolic expression of the term, but as a symbolic functional. Only contains symbols."""
        foo = disable_commutativity(self.symbolic_expression)
        ss = disable_commutativity(self.field.symbol)
        return Lambda(ss, foo)

    @property
    def numerical_function(self):
        """~sympy.core.expr.Expr: The numeric expression of the term, as a symbolic functional,
        but with parameters replaced by their numerical value.
        Same as the symbolic function for this term."""
        foo = disable_commutativity(self.numerical_expression)
        ss = disable_commutativity(self.field.symbol)
        return Lambda(ss, foo)

    @property
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: The symbolic expression of the term. Only contains symbols."""
        return sproduct(self.sign, self.field.symbol)

    @property
    def numerical_expression(self):
        """~sympy.core.expr.Expr: The numeric expression of the term, with parameters replaced by their numerical value.
        Same as the symbolic expression for this term."""
        return self.symbolic_expression

    @property
    def latex(self):
        """str: Return a LaTeX representation of the term."""
        if self.sign > 0:
            s = '+ '
        else:
            s = '- '
        return s + self.field.latex

    def _inner_product_arguments(self, basis, indices, numerical=False):
        pass

    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        """Return nor compute nothing. Not used by this class."""
        pass
