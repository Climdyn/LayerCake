

from sympy import Symbol, S, Eq


class Equation(object):

    _t = Symbol('t')

    def __init__(self, field, other_fields=None):

        self.field = field
        self.other_fields = other_fields
        self.terms = list()

    def add_term(self, term):

        self.terms.append(term)

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

