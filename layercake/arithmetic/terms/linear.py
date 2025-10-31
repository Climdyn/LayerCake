
from layercake.arithmetic.terms.base import SingleArithmeticTerm
from layercake.arithmetic.utils import sproduct


class LinearTerm(SingleArithmeticTerm):

    def __init__(self, field, inner_product_definition=None, prefactor=None, name='', sign=1):

        SingleArithmeticTerm.__init__(self, field, inner_product_definition, prefactor, name, sign=sign)

    @property
    def symbolic_expression(self):
        if self.prefactor is None:
            return sproduct(self.sign, self.field.symbol)
        else:
            return sproduct(self.sign * self.prefactor.symbol, self.field.symbol)

    @property
    def numerical_expression(self):
        if self.prefactor is None:
            return sproduct(self.sign, self.field.symbol)
        else:
            return sproduct(self.sign * self.prefactor, self.field.symbol)
