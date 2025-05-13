
from layercake.arithmetic.terms.base import OperationOnTerms
from layercake.arithmetic.utils import sproduct, sadd


class ProductOfTerms(OperationOnTerms):

    def __init__(self, *terms, name='', rank=None, sign=1):

        OperationOnTerms.__init__(self, *terms, name=name, rank=rank, sign=sign)

    def operation(self, *terms, evaluate=False):
        return sproduct(*terms, evaluate=evaluate)

    def _compute_rank(self):
        self._rank = 1
        for term in self._terms:
            self._rank += term.rank - 1


class AdditionOfTerms(OperationOnTerms):

    def __init__(self, *terms, name='', rank=None, sign=1):

        OperationOnTerms.__init__(self, *terms, name=name, rank=rank, sign=sign)

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

    def operation(self, *terms, evaluate=False):
        return sadd(*terms, evaluate=evaluate)
