

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
    def expression(self):
        rterm = S.Zero
        for term  in self.terms:
            rterm += term.expression
        return Eq(self.field.symbol.diff(self._t), rterm)

    @property
    def rhs(self):
        rterm = S.Zero
        for term  in self.terms:
            rterm += term.expression
        return rterm

    @property
    def lhs(self):
        return self.field.symbol.diff(self._t)

