

from sympy import Symbol, S, Eq
from layercake.arithmetic.terms.base import ArithmeticTerm


class Equation(object):

    _t = Symbol('t')

    def __init__(self, field, other_fields=None):

        self.field = field
        self.other_fields = other_fields
        self.terms = list()

    def add_term(self, term):
        if not issubclass(term.__class__, ArithmeticTerm):
            raise ValueError('Provided term must be a valid ArithmeticTerm object.')
        self.terms.append(term)

    def add_terms(self, terms):
        for t in terms:
            self.add_term(t)

    @property
    def symbolic_expression(self):
        rterm = S.Zero
        for term in self.terms:
            rterm += term.symbolic_expression
        return Eq(self.field.function.diff(self._t), rterm)

    @property
    def numerical_expression(self):
        rterm = S.Zero
        for term in self.terms:
            rterm += term.numerical_expression
        return Eq(self.field.function.diff(self._t), rterm)

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
    def lhs(self):
        return self.field.symbol.diff(self._t)

