
from layercake.arithmetic.terms.base import OperationOnTerms
from layercake.arithmetic.utils import sproduct


class ProductOfTerms(OperationOnTerms):

    def __init__(self, *terms, name='', rank=None, sign=1):

        if len(terms) < 2:
            raise ValueError('More than one term must be provided to this class.')

        OperationOnTerms.__init__(self, *terms, name=name, rank=rank, sign=sign)

    def operation(self, *terms, evaluate=False):
        return sproduct(*terms, evaluate=evaluate)

