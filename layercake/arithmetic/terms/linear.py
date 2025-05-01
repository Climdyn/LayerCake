
from layercake.arithmetic.terms.base import SingleArithmeticTerm
from layercake.utils.commutativity import disable_commutativity
from sympy import Mul


class LinearTerm(SingleArithmeticTerm):

    def __init__(self, field, inner_product_definition=None, prefactor=None, name=''):

        SingleArithmeticTerm.__init__(self, field, inner_product_definition, prefactor, name)

    @property
    def symbolic_expression(self):
        if self.prefactor is None:
            return self.field.symbol
        else:
            return Mul(self.prefactor.symbol, self.field.symbol, evaluate=False)

    @property
    def numerical_expression(self):
        if self.prefactor is None:
            return self.field.symbol
        else:
            return Mul(self.prefactor, self.field.symbol, evaluate=False)
