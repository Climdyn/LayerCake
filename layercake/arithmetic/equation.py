
"""

    Equation definition module
    ==========================

    Main class to define partial differential equation.
    This class is the workhorse of LayerCake to define and specify
    partial differential equation.

"""

from sympy import Symbol, S, Eq
import matplotlib.pyplot as plt
from layercake.arithmetic.terms.base import ArithmeticTerms
from layercake.utils import isin


# TODO: deal with the cases where lists of terms are empty - maybe nothing is needed but to check

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
    lhs_terms: ~arithmetic.terms.base.ArithmeticTerms or list(~arithmetic.terms.base.ArithmeticTerms), optional
        Terms on the left-hand side of the equation. At least one must involve the field defined above.
    name: str, optional
        Name for the equation.

    Attributes
    ----------
    field: ~field.Field
        The spatial field over which the partial differential equation.
    rhs_terms: list(~arithmetic.terms.base.ArithmeticTerms)
        List of additive terms in the right-hand side of the equation.
    lhs_terms: ListOfArithmeticTerms(~arithmetic.terms.base.ArithmeticTerms)
        Term on the left-hand side of the equation.
    name: str
        Optional name for the equation.
    """

    _t = Symbol('t')

    def __init__(self, field, lhs_terms=None, name=''):

        self.field = field
        self.field._equation = self
        self.rhs_terms = ListOfAdditiveArithmeticTerms()
        if lhs_terms is None:
            self.lhs_terms = ListOfAdditiveArithmeticTerms()
        elif isinstance(lhs_terms, list):
            self.lhs_terms = ListOfAdditiveArithmeticTerms(lhs_terms)
        else:
            self.lhs_terms = ListOfAdditiveArithmeticTerms([lhs_terms])
        self.name = name
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

    def add_lhs_term(self, term):
        """Add a term to the left-hand side of the equation.

        Parameters
        ----------
        term: ~arithmetic.terms.base.ArithmeticTerms
            Term to be added to the left-hand side of the equation.
        """
        if not issubclass(term.__class__, ArithmeticTerms):
            raise ValueError('Provided term must be a valid ArithmeticTerm object.')
        self.lhs_terms.append(term)

    def add_lhs_terms(self, terms):
        """Add multiple terms to the left-hand side of the equation.

        Parameters
        ----------
        terms: list(~arithmetic.terms.base.ArithmeticTerms)
            Terms to be added to the left-hand side of the equation.
        """
        for t in terms:
            self.add_lhs_term(t)

    @property
    def terms(self):
        """Alias for the list of RHS arithmetic terms."""
        return self.rhs_terms

    @property
    def other_fields(self):
        """list(~field.Field): List of additional fields present in the equation."""
        other_fields = list()
        for equation_term in self.terms:
            for term in equation_term.terms:
                if term.field is not self.field and term.field.dynamical and term.field not in other_fields:
                    other_fields.append(term.field)
        other_fields = other_fields + self.other_fields_in_lhs
        return other_fields

    @property
    def other_fields_in_lhs(self):
        """list(~field.Field): List of additional fields present in the LHS of the equation."""
        other_fields = list()
        for equation_term in self.lhs_terms:
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
        for equation_term in self.lhs_terms:
            for term in equation_term.terms:
                if term.field is not self.field and not term.field.dynamical and term.field not in parameter_fields:
                    parameter_fields.append(term.field)
        return parameter_fields

    @property
    def parameters(self):
        """list(~parameter.Parameter): List of parameters present in the equation."""
        parameters_list = list()
        for term in self.terms + self.lhs_terms:
            params_list = term.parameters
            for param in params_list:
                if not isin(param, parameters_list):
                    parameters_list.append(param)

        for param_field in self.parameter_fields:
            for param in param_field.parameters:
                if param is not None and not isin(param, parameters_list):
                    parameters_list.append(param)

        return parameters_list

    @property
    def parameters_symbols(self):
        """list(~sympy.core.symbol.Symbol): List of parameter's symbols present in the equation."""
        return [p.symbol for p in self.parameters]

    @property
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: Symbolic expression of the equation."""
        return Eq(self.symbolic_lhs.diff(self._t, evaluate=False), self.rhs_terms.symbolic_expression)

    @property
    def numerical_expression(self):
        """~sympy.core.expr.Expr: Expression of the equation with parameters replaced by their
        configured values."""
        return Eq(self.numerical_lhs.diff(self._t, evaluate=False), self.rhs_terms.symbolic_expression)

    @property
    def symbolic_rhs(self):
        """~sympy.core.expr.Expr: Symbolic expression of the right-hand side of the equation."""
        return self.rhs_terms.symbolic_expression

    @property
    def numerical_rhs(self):
        """~sympy.core.expr.Expr: Expression of the right-hand side of the equation with
        parameters replaced by their configured values."""
        return self.rhs_terms.numerical_expression

    @property
    def symbolic_lhs(self):
        """~sympy.core.expr.Expr: Symbolic expression of the left-hand side of the equation."""
        return self.lhs_terms.symbolic_expression

    @property
    def numerical_lhs(self):
        """~sympy.core.expr.Expr: Expression of the left-hand side of the equation with
        parameters replaced by their configured values."""
        return self.lhs_terms.numerical_expression

    @property
    def lhs_inner_products(self):
        """list(~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.tensor.array.ImmutableSparseNDimArray) or list(sparse.COO(float)): Inner products of each term of
        the left-hand side of the equation, if available."""
        return [term.inner_products for term in self.lhs_terms]

    @property
    def lhs_inner_products_addition(self):
        """~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.tensor.array.ImmutableSparseNDimArray or sparse.COO(float): Added left-hand
        side inner products of the equation, if available. Might raise an error if not all terms are compatible."""
        result = self.lhs_terms[0].inner_products.copy()
        for term in self.lhs_terms[1:]:
            result = result + term.inner_products
        return result

    @property
    def maximum_rank(self):
        """int: Maximum rank of the right-hand side terms tensors."""
        return max(self.terms.maximum_rank, self.lhs_terms.maximum_rank)

    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        """Compute the inner products tensor of the left-hand and right-hand side terms.

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
            Control the switch from symbolic to numerical integration. By default, `parallel_integration` workers will try to integrate
            |Sympy| expressions symbolically, but a fallback to numerical integration can be enforced.
            The options are:

            * `None`: This is the "full-symbolic" mode. No timeout will be applied, and the switch to numerical integration will never happen.
              Can result in very long and improbable computation time.
            * `True`: This is the "full-numerical" mode. Symbolic computations do not occur, and the workers try directly to integrate
              numerically.
            * `False`: Same as `None`.
            * An integer: defines a timeout after which, if a symbolic integration have not completed, the worker switch to the
              numerical integration.
        permute: bool, optional
            If `True`, applies all the possible permutations to the tensor indices
            from 1 to the rank of the tensor.
            Default to `False`, i.e. no permutation is applied.
        """
        self.lhs_terms.compute_inner_products(basis, numerical, timeout, num_threads, permute)
        self.rhs_terms.compute_inner_products(basis, numerical, timeout, num_threads, permute)

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

        Returns
        -------
        str
            The LaTeX string representing the equation.
        """
        lhs = self.lhs_terms[0].latex
        for term in self.lhs_terms[1:]:
            lhs += term.latex
        if drop_first_lhs_char:
            lhs = lhs[2:]
        if enclose_lhs:
            latex_string = r'\frac{\partial}{\partial t} ' + r'\left(' + lhs + r'\right)'
        else:
            latex_string = r'\frac{\partial}{\partial t} ' + lhs

        first_term = self.terms[0].latex
        if drop_first_rhs_char:
            first_term = first_term[2:]
        latex_string += ' = ' + first_term

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


class ListOfAdditiveArithmeticTerms(list):
    """Class holding list of additive arithmetic terms in equations."""

    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        """Compute the inner products tensor of the all the terms of the list.

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
            Control the switch from symbolic to numerical integration. By default, `parallel_integration` workers will try to integrate
            |Sympy| expressions symbolically, but a fallback to numerical integration can be enforced.
            The options are:

            * `None`: This is the "full-symbolic" mode. No timeout will be applied, and the switch to numerical integration will never happen.
              Can result in very long and improbable computation time.
            * `True`: This is the "full-numerical" mode. Symbolic computations do not occur, and the workers try directly to integrate
              numerically.
            * `False`: Same as `None`.
            * An integer: defines a timeout after which, if a symbolic integration have not completed, the worker switch to the
              numerical integration.
        permute: bool, optional
            If `True`, applies all the possible permutations to the tensor indices
            from 1 to the rank of the tensor.
            Default to `False`, i.e. no permutation is applied.
        """

        for term in self:
            term.compute_inner_products(basis, numerical, timeout, num_threads, permute)

    @property
    def maximum_rank(self):
        """int: Maximum rank of the right-hand side terms tensors."""
        max_rank = 0
        for term in self:
            max_rank = max(max_rank, term.rank)
        return max_rank

    @property
    def same_rank(self):
        """bool: Check if all terms have the same rank."""
        rank = self[0].rank
        for term in self[1:]:
            if term.rank != rank:
                return False
        return True

    @property
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: Symbolic expression of the collection of additive arithmetic terms."""

        rterm = S.Zero
        for term in self:
            rterm += term.symbolic_expression
        return rterm

    @property
    def numerical_expression(self):
        """~sympy.core.expr.Expr: Expression of the collection of additive arithmetic terms with
        parameters replaced by their configured values."""

        rterm = S.Zero
        for term in self:
            rterm += term.numerical_expression
        return rterm
