
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

    @property
    def latex(self):
        if self.sign > 0:
            s = f'+ '
        else:
            s = f'- '
        if self.prefactor is None:
            return s + self.field.latex
        if hasattr(self.prefactor, 'latex'):
            if self.prefactor.latex is not None:
                s += f'{self.prefactor.latex} '
                return s + self.field.latex
        if hasattr(self.prefactor, 'symbol'):
            if self.prefactor.symbol is not None:
                s += f'{self.prefactor.symbol} '
                return s + self.field.latex

        s += f'{self.prefactor} '
        return s + self.field.latex
