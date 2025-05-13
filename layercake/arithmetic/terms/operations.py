
from layercake.arithmetic.terms.base import OperationOnTerms
from layercake.arithmetic.utils import sproduct, sadd


class ProductOfTerms(OperationOnTerms):

    def __init__(self, *terms, name='', rank=None, sign=1):

        OperationOnTerms.__init__(self, *terms, name=name, rank=rank, sign=sign)

    def operation(self, *terms, evaluate=False):
        return sproduct(*terms, evaluate=evaluate)


class AdditionOfTerms(OperationOnTerms):

    def __init__(self, *terms, name='', rank=None, sign=1):

        OperationOnTerms.__init__(self, *terms, name=name, rank=rank, sign=sign)

    def operation(self, *terms, evaluate=False):
        return sadd(*terms, evaluate=evaluate)
