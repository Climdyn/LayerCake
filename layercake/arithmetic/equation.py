
"""

    Equation definition module
    ==========================

    Main class to define partial differential equation.
    This class is the workhorse of LayerCake to define and specify
    partial differential equation.

"""

from sympy import Symbol, S, Eq
from layercake.arithmetic.terms.base import ArithmeticTerms, OperationOnTerms
import matplotlib.pyplot as plt


class Equation(object):
    """Main class to define and specify partial differential equations.

    Notes
    -----
    The left-hand side is always expressed as a partial time derivative of
    something: :math:`\\partial_t` .

    Parameters
    ----------
    field: ~field.Field
        The spatial field over which the partial differential equation.
    lhs_term: ~arithmetic.terms.base.ArithmeticTerms
        Term on the left-hand side of the equation.
        Must be a single term, possibly a combination
        through :class:`~arithmetic.terms.base.OperationOnTerms` operations.

    Attributes
    ----------
    field: ~field.Field
        The spatial field over which the partial differential equation.
    terms: list(~arithmetic.terms.base.ArithmeticTerms)
        List of additive terms in the right-hand side of the equation.
    lhs_term: ~arithmetic.terms.base.ArithmeticTerms
        Term on the left-hand side of the equation.
    """

    _t = Symbol('t')

    def __init__(self, field, lhs_term):

        self.field = field
        self.field._equation = self
        self.terms = list()
        self.lhs_term = lhs_term
        self.lhs_term.field = self.field
        self._layer = None
        self._cake = None

    def add_rhs_term(self, term):
        """Add a term to the right-hand side of the equation.

        Parameters
        ----------
        term: ~arithmetic.terms.base.ArithmeticTerms
            Term to be added to the right-hand side of the equation.
        """
        if not issubclass(term.__class__, ArithmeticTerms):
            raise ValueError('Provided term must be a valid ArithmeticTerm object.')
        self.terms.append(term)

    def add_rhs_terms(self, terms):
        """Add multiple terms to the right-hand side of the equation.

        Parameters
        ----------
        terms: list(~arithmetic.terms.base.ArithmeticTerms)
            Terms to be added to the right-hand side of the equation.
        """
        for t in terms:
            self.add_rhs_term(t)

    @property
    def other_fields(self):
        """list(~field.Field): List of additional fields present in the equation."""
        other_fields = list()
        for equation_term in self.terms:
            for term in equation_term.terms:
                if term.field is not self.field and term.field.dynamical and term.field not in other_fields:
                    other_fields.append(term.field)
        return other_fields

    @property
    def parameter_fields(self):
        """list(~field.ParameterField): List of non-dynamical parameter fields present in the equation."""
        parameter_fields = list()
        for equation_term in self.terms:
            for term in equation_term.terms:
                if term.field is not self.field and not term.field.dynamical and term.field not in parameter_fields:
                    parameter_fields.append(term.field)
        for term in self.lhs_term.terms:
            if term.field is not self.field and not term.field.dynamical and term.field not in parameter_fields:
                parameter_fields.append(term.field)
        return parameter_fields

    @property
    def parameters(self):
        """list(~field.Parameter): List of parameters present in the equation."""
        parameters_list = list()
        for term in self.terms + [self.lhs_term]:
            if issubclass(term.__class__, OperationOnTerms):
                for tterm in term.terms:
                    param = tterm.prefactor
                    if param is not None and not self._isin(param, parameters_list):
                        parameters_list.append(param)
            else:
                param = term.prefactor
                if param is not None and not self._isin(param, parameters_list):
                    parameters_list.append(param)

        for param_field in self.parameter_fields:
            for param in param_field.parameters:
                if param is not None and not self._isin(param, parameters_list):
                    parameters_list.append(param)

        return parameters_list

    @staticmethod
    def _isin(o, it):
        res = False
        for i in it:
            if o is i:
                res = True
                break
        return res

    @property
    def parameters_symbols(self):
        """list(~sympy.core.symbol.Symbol): List of parameter's symbols present in the equation."""
        return [p.symbol for p in self.parameters]

    @property
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: Symbolic expression of the equation."""
        rterm = S.Zero
        for term in self.terms:
            rterm += term.symbolic_expression
        return Eq(self.symbolic_lhs.diff(self._t, evaluate=False), rterm)

    @property
    def numerical_expression(self):
        """~sympy.core.expr.Expr: Expression of the equation with parameters replaced by their
        configured values."""
        rterm = S.Zero
        for term in self.terms:
            rterm += term.numerical_expression
        return Eq(self.numerical_lhs.diff(self._t, evaluate=False), rterm)

    @property
    def symbolic_rhs(self):
        """~sympy.core.expr.Expr: Symbolic expression of the right-hand side of the equation."""
        rterm = S.Zero
        for term in self.terms:
            rterm += term.symbolic_expression
        return rterm

    @property
    def numerical_rhs(self):
        """~sympy.core.expr.Expr: Expression of the right-hand side of the equation with
        parameters replaced by their configured values."""
        rterm = S.Zero
        for term in self.terms:
            rterm += term.numerical_expression
        return rterm

    @property
    def symbolic_lhs(self):
        """~sympy.core.expr.Expr: Symbolic expression of the left-hand side of the equation."""
        return self.lhs_term.symbolic_expression

    @property
    def numerical_lhs(self):
        """~sympy.core.expr.Expr: Expression of the left-hand side of the equation with
        parameters replaced by their configured values."""
        return self.lhs_term.numerical_expression

    @property
    def lhs_inner_products(self):
        """~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.tensor.array.ImmutableSparseNDimArray or sparse.COO(float): Left-hand
        side inner products of the equation, if available."""
        return self.lhs_term.inner_products

    @property
    def maximum_rank(self):
        """int: Maximum rank of the right-hand side terms tensors."""
        rhs_max_rank = 0
        for term in self.terms:
            rhs_max_rank = max(rhs_max_rank, term.rank)
        return max(rhs_max_rank, self.lhs_term.rank)

    def compute_lhs_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        """Compute the inner products tensor of the left-hand side term.

        Parameters
        ----------
        basis: SymbolicBasis
            Basis with which to compute the inner products.
        numerical: bool, optional
            Whether the resulting computed inner products must be numerical or symbolic.
            Default to `False`, i.e. symbolic output.
        num_threads: int or None, optional
            Number of threads to use to compute the inner products. If `None` use all the cpus available.
            Default to `None`.
        timeout: int or float or bool or None, optional
            The timeout for the numerical computation of each inner product.
            After the timeout, compute the inner product with a quadrature instead of symbolic integration.
            Does not apply to symbolic output computations.
            If `None` or `False`, no timeout occurs.
            Default to `None`.
        permute: bool, optional
            If `True`, applies all the possible permutations to the tensor indices
            from 1 to the rank of the tensor.
            Default to `False`, i.e. no permutation is applied.
        """
        self.lhs_term.compute_inner_products(basis, numerical, timeout, num_threads, permute)

    def to_latex(self, enclose_lhs=True, drop_first_lhs_char=True, drop_first_rhs_char=False):
        """Generate a LaTeX string representing the equation mathematically.

        Parameters
        ----------
        enclose_lhs: bool, optional
            Whether to enclose the left-hand side term inside parenthesis.
            Default to `True`.
        drop_first_lhs_char: bool, optional
            Whether to drop the first two character of the left-hand side latex string.
            Useful to drop the sign in front of it.
            Default to `True`.
        drop_first_rhs_char: bool, optional
            Whether to drop the first two character of the right-hand side latex string.
            Useful to drop the sign in front of it.
            Default to `False`.
        """
        lhs = self.lhs_term.terms[0].latex
        if drop_first_lhs_char:
            lhs = lhs[2:]
        if enclose_lhs:
            latex_string = r'\frac{\partial}{\partial t} ' + r'\left(' + lhs + r'\right)'
        else:
            latex_string = r'\frac{\partial}{\partial t} ' + lhs

        fterm = self.terms[0].latex
        if drop_first_rhs_char:
            fterm = fterm[2:]
        latex_string += ' = ' + fterm

        for term in self.terms[1:]:
            latex_string += term.latex

        return latex_string

    def show_latex(self, enclose_lhs=True, drop_first_lhs_char=True, drop_first_rhs_char=False):
        """Show the LaTeX string representing the equation mathematically rendered in a window.

        Parameters
        ----------
        enclose_lhs: bool, optional
            Whether to enclose the left-hand side term inside parenthesis.
            Default to `True`.
        drop_first_lhs_char: bool, optional
            Whether to drop the first two character of the left-hand side latex string.
            Useful to drop the sign in front of it.
            Default to `True`.
        drop_first_rhs_char: bool, optional
            Whether to drop the first two character of the right-hand side latex string.
            Useful to drop the sign in front of it.
            Default to `False`.
        """
        latex_string = self.to_latex(enclose_lhs=enclose_lhs,
                                     drop_first_lhs_char=drop_first_lhs_char,
                                     drop_first_rhs_char=drop_first_rhs_char
                                     )
        plt.figure(figsize=(8, 2))
        plt.axis('off')
        plt.text(-0.1, 0.5, '$%s$' % latex_string)
        plt.show()

    def __repr__(self):
        eq = self.symbolic_expression
        return f'{eq.lhs} = {eq.rhs}'

    def __str__(self):
        return self.__repr__()
