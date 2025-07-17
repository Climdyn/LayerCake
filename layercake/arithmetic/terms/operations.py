
from layercake.arithmetic.terms.base import OperationOnTerms
from layercake.arithmetic.utils import sproduct, sadd


class ProductOfTerms(OperationOnTerms):

    def __init__(self, *terms, name='', rank=None, sign=1):

        OperationOnTerms.__init__(self, *terms, name=name, rank=rank, sign=sign)

    def _compute_rank(self):
        self._rank = 1
        for term in self._terms:
            self._rank += term.rank - 1

    def _create_inner_products_basis_list(self, basis):
        basis_list = [basis]
        for term in self._terms:
            for _ in range(term.rank - 1):
                basis_list.append(term.field.basis)

        return basis_list

    def operation(self, *terms, evaluate=False):
        return sproduct(*terms, evaluate=evaluate)


class AdditionOfTerms(OperationOnTerms):

    def __init__(self, *terms, name='', rank=None, sign=1):

        OperationOnTerms.__init__(self, *terms, name=name, rank=rank, sign=sign)

        for i, term1 in enumerate(terms):
            for term2 in terms[i:]:
                if term2.field is not term1.field:
                    raise ValueError(f'AdditionOfTerms must always involve the same field, '
                                     f'but two different fields {term1.field} and {term2.field} have been provided.')

    @property
    def symbolic_expression(self):
        return sproduct(self.sign, self.operation(*map(lambda t: t.symbolic_expression, self._terms)))

    @property
    def numerical_expression(self):
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
        return basis, self._terms[0].field.basis

    def operation(self, *terms, evaluate=False):
        return sadd(*terms, evaluate=evaluate)
