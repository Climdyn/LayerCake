
"""

    Arithmetic operations module
    ============================

    This module defines terms resulting from arithmetic operation being applied to arithmetic terms.
    The corresponding objects hold the symbolic representation of the terms and their decomposition
    on given function basis.

    Description of the classes
    --------------------------

    * :class:`ProductOfTerms`: Product of a list of terms in a partial differential equation.
    * :class:`AdditionOfTerms`: Addition of a list of terms in a partial differential equation.

"""

from layercake.arithmetic.terms.base import OperationOnTerms
from layercake.arithmetic.utils import sproduct, sadd


class ProductOfTerms(OperationOnTerms):
    """Term representing the product of multiple arithmetic terms.

    Parameters
    ----------
    *terms: ArithmeticTerms
        Arithmetic terms to take the product of.
    name: str, optional
        Name of the term.
    rank: int, optional
        Can be used to force the rank of the term, i.e. force the rank of the tensor storing the term(s) decomposition on the provided function basis.
        Compute the rank automatically if not provided.
    sign: int, optional
        Sign in front of the term. Either +1 or -1.
        Default to +1.

    Attributes
    ----------
    name: str
        Name of the term.
    sign: int
        Sign in front of the term. Either +1 or -1.
    inner_products: None or ~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.tensor.array.ImmutableSparseNDimArray or sparse.COO(float)
        The inner products tensor of the term.
        Set initially to `None` (not computed).
    inner_product_definition: InnerProductDefinition
        Object defining the integral representation of the inner product that is used to compute the term representation
        on a given function basis.
    """

    def __init__(self, *terms, name='', rank=None, sign=1):

        OperationOnTerms.__init__(self, *terms, name=name, rank=rank, sign=sign)

    def _compute_rank(self):
        """Routine to compute the rank of the inner products tensor, if not enforced by the user.

        Returns
        -------
        int
            The rank of the inner products tensor.
        """
        self._rank = 1
        for term in self._terms:
            self._rank += term.rank - 1

    def _create_inner_products_basis_list(self, basis):
        """Function defining the list of symbolic function basis specified in order to compute the inner products
        for the operation on terms. Must be defined in subclasses.

        Parameters
        ----------
        basis: SymbolicBasis
            Basis to put on the left-hand side of the inner products.

        Returns
        -------
        tuple(SymbolicBasis)
            List of symbolic function basis used to compute the inner products.

        """
        basis_list = [basis]
        for term in self._terms:
            for _ in range(term.rank - 1):
                basis_list.append(term.field.basis)

        return tuple(basis_list)

    def operation(self, *terms, evaluate=False):
        """Operation (here the product) acting on the terms.

        Parameters
        ----------
        *terms: ArithmeticTerms
            Terms on which the operation must be applied.
        evaluate: bool
            Whether to let `Sympy`_ evaluate the operation or not.
            Default to `False`.

        Returns
        -------
        ~sympy.core.expr.Expr
            The result of the operation on the terms, as a `Sympy`_ symbolic expression.

        .. _Sympy: https://www.sympy.org/
        """
        return sproduct(*terms, evaluate=evaluate)

    @property
    def latex(self):
        """str: Return a LaTeX representation of the terms."""
        if self.sign > 0:
            s = f'+ '
        else:
            s = f'- '

        latexes = [t.latex for t in self.terms]
        for lat in latexes:
            if lat[0] == '-':
                s += f'({lat}) '
            else:
                s += f'{lat[1:]} '
        return s


class AdditionOfTerms(OperationOnTerms):
    """Term representing the addition of multiple arithmetic terms.

    Warnings
    --------
    Provided terms must have the same rank, and involve the same field.

    Parameters
    ----------
    *terms: ArithmeticTerms
        Arithmetic terms to take the addition of.
    name: str, optional
        Name of the term.
    rank: int, optional
        Can be used to force the rank of the term, i.e. force the rank of the tensor storing the term(s) decomposition on the provided function basis.
        Compute the rank automatically if not provided.
    sign: int, optional
        Sign in front of the term. Either +1 or -1.
        Default to +1.

    Attributes
    ----------
    name: str
        Name of the term.
    sign: int
        Sign in front of the term. Either +1 or -1.
    inner_products: None or ~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.tensor.array.ImmutableSparseNDimArray or sparse.COO(float)
        The inner products tensor of the term.
        Set initially to `None` (not computed).
    inner_product_definition: InnerProductDefinition
        Object defining the integral representation of the inner product that is used to compute the term representation
        on a given function basis.
    """

    def __init__(self, *terms, name='', rank=None, sign=1):

        OperationOnTerms.__init__(self, *terms, name=name, rank=rank, sign=sign)

        for i, term1 in enumerate(terms):
            for term2 in terms[i:]:
                if term2.field is not term1.field:
                    raise ValueError(f'AdditionOfTerms must always involve the same field, '
                                     f'but two different fields {term1.field} and {term2.field} have been provided.')

    @property
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: The symbolic expression of the result of the operation on the terms, but as a symbolic functional.
        Only contains symbols."""
        return sproduct(self.sign, self.operation(*map(lambda t: t.symbolic_expression, self._terms)))

    @property
    def numerical_expression(self):
        """~sympy.core.expr.Expr: The numerical expression of the result of the operation on the terms, as a symbolic functional,
        and with parameters replaced by their numerical value."""
        return sproduct(self.sign, self.operation(*map(lambda t: t.numerical_expression, self._terms)))

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

    def _compute_rank(self):
        """Routine to compute the rank of the inner products tensor, if not enforced by the user.

        Returns
        -------
        int
            The rank of the inner products tensor.
        """
        self._rank = 0
        prank = 0
        for i, term in enumerate(self._terms):
            trank = term.rank
            if i > 0:
                if trank != prank:
                    raise ValueError('All the terms provided to AdditionOfTerms term must be of the same rank.')
            else:
                self._rank = trank

            prank = term.rank

    def _create_inner_products_basis_list(self, basis):
        """Function defining the list of symbolic function basis specified in order to compute the inner products
        for the operation on terms. Must be defined in subclasses.

        Parameters
        ----------
        basis: SymbolicBasis
            Basis to put on the left-hand side of the inner products.

        Returns
        -------
        tuple(SymbolicBasis)
            List of symbolic function basis used to compute the inner products.

        """
        return basis, self._terms[0].field.basis

    def operation(self, *terms, evaluate=False):
        """Operation (here the addition) acting on the terms.

        Parameters
        ----------
        *terms: ArithmeticTerms
            Terms on which the operation must be applied.
        evaluate: bool
            Whether to let `Sympy`_ evaluate the operation or not.
            Default to `False`.

        Returns
        -------
        ~sympy.core.expr.Expr
            The result of the operation on the terms, as a `Sympy`_ symbolic expression.

        .. _Sympy: https://www.sympy.org/
        """
        return sadd(*terms, evaluate=evaluate)

    @property
    def latex(self):
        """str: Return a LaTeX representation of the terms."""
        latexes = [t.latex for t in self.terms]
        s = ''
        for lat in latexes:
            s += f'{lat} '
        return s
