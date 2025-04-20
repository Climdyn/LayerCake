
from layercake.arithmetic.terms.base import ArithmeticTerm


class LinearTerm(ArithmeticTerm):

    def __init__(self, field, factor):

        self.name = 'Linear term'
        self.inner_products = None
        self.field = field
        self.factor = factor

    @property
    def expression(self):
        return self.factor * self.field.symbol
