
from layercake.arithmetic.terms.base import OperationOnTerms
from layercake.utils.commutativity import disable_commutativity
from sympy import Mul


class ProductOfTerms(OperationOnTerms):

    def __init__(self, *terms, name='', rank=None):

        if len(terms) < 2:
            raise ValueError('More than one term must be provided to this class.')

        OperationOnTerms.__init__(self, *terms, name=name, rank=rank)

    def operation(self, *terms, evaluate=False):
        for i, t in enumerate(terms):
            if i == 0:
                res = t
            else:
                res = Mul(res, t, evaluate=evaluate)
        return res

