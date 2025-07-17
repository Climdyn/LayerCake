

from sympy import Symbol, S, Eq
from layercake.arithmetic.terms.base import ArithmeticTerms, OperationOnTerms


class Equation(object):

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
        if not issubclass(term.__class__, ArithmeticTerms):
            raise ValueError('Provided term must be a valid ArithmeticTerm object.')
        self.terms.append(term)

    def add_rhs_terms(self, terms):
        for t in terms:
            self.add_rhs_term(t)

    @property
    def other_fields(self):
        other_fields = list()
        for equation_term in self.terms:
            for term in equation_term.terms:
                if term.field is not self.field and term.field.dynamical and term.field not in other_fields:
                    other_fields.append(term.field)
        return other_fields

    @property
    def parameter_fields(self):
        parameter_fields = list()
        for equation_term in self.terms:
            for term in equation_term.terms:
                if term.field is not self.field and not term.field.dynamical and term.field not in parameter_fields:
                    parameter_fields.append(term.field)
        return parameter_fields

    @property
    def parameters(self):
        parameters_list = list()
        for term in self.terms:
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
        return [p.symbol for p in self.parameters]

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

    def __repr__(self):
        eq = self.symbolic_expression
        return f'{eq.lhs} = {eq.rhs}'

    def __str__(self):
        return self.__repr__()
