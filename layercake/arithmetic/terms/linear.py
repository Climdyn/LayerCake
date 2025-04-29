
from layercake.arithmetic.terms.base import SingleArithmeticTerm
from layercake.utils.commutativity import disable_commutativity
from sympy import Mul


class LinearTerm(SingleArithmeticTerm):

    def __init__(self, field, inner_product_definition, parameter=None, name=''):

        SingleArithmeticTerm.__init__(self, field, inner_product_definition, name)
        self.parameter = parameter
        self._rank = 1

    @property
    def symbolic_expression(self):
        if self.parameter is None:
            return self.field.symbol
        else:
            return Mul(self.parameter.symbol, self.field.symbol, evaluate=False)

    @property
    def numerical_expression(self):
        if self.parameter is None:
            return self.field.symbol
        else:
            return Mul(self.parameter, self.field.symbol, evaluate=False)
