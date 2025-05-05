

from sympy import Symbol, S, Eq, Mul
from layercake.arithmetic.terms.base import ArithmeticTerm


class Equation(object):

    _t = Symbol('t')

    def __init__(self, field, lhs_term,  inner_product_definition=None, other_fields=None, lhs_prefactor=None):

        self.field = field
        self.other_fields = other_fields
        self.terms = list()
        self.lhs_term = lhs_term(field, inner_product_definition, lhs_prefactor, 'Left Hand Side')
        self._layer = None
        self._cake = None

    def add_rhs_term(self, term):
        if not issubclass(term.__class__, ArithmeticTerm):
            raise ValueError('Provided term must be a valid ArithmeticTerm object.')
        self.terms.append(term)

    def add_rhs_terms(self, terms):
        for t in terms:
            self.add_rhs_term(t)

    @property
    def symbolic_expression(self):
        rterm = S.Zero
        for term in self.terms:
            rterm += term.symbolic_expression
        return Eq(self.symbolic_lhs.diff(self._t, evaluate=False), rterm)

    @property
    def numerical_expression(self):
        rterm = S.Zero
        for term in self.terms:
            rterm += term.numerical_expression
        return Eq(self.numerical_lhs.diff(self._t, evaluate=False), rterm)

    @property
    def symbolic_rhs(self):
        rterm = S.Zero
        for term in self.terms:
            rterm += term.symbolic_expression
        return rterm

    @property
    def numerical_rhs(self):
        rterm = S.Zero
        for term in self.terms:
            rterm += term.numerical_expression
        return rterm

    @property
    def symbolic_lhs(self):
        return self.lhs_term.symbolic_expression

    @property
    def numerical_lhs(self):
        return self.lhs_term.numerical_expression

    @property
    def lhs_inner_products(self):
        return self.lhs_term.inner_products

    @property
    def maximum_rank(self):
        rhs_max_rank = 0
        for term in self.terms:
            rhs_max_rank = max(rhs_max_rank, term.rank)
        return max(rhs_max_rank, self.lhs_term.rank)

    def compute_lhs_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        self.lhs_term.compute_inner_products(basis, numerical, timeout, num_threads, permute)

