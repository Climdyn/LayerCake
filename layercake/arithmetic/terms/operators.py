
from layercake.arithmetic.terms.base import SingleArithmeticTerm
from layercake.arithmetic.utils import sproduct


class OperatorTerm(SingleArithmeticTerm):

    def __init__(self, field, operator, operator_args, inner_product_definition=None, prefactor=None, name='', sign=1):

        SingleArithmeticTerm.__init__(self, field, inner_product_definition, prefactor, name, sign=sign)
        if not isinstance(operator_args, (tuple, list)):
            operator_args = tuple([operator_args])
        self._operator = operator(*operator_args)

    @property
    def symbolic_expression(self):
        if self.prefactor is None:
            return sproduct(self.sign * self._operator, self.field.symbol)
        else:
            return sproduct(self.sign * self.prefactor.symbol, self._operator, self.field.symbol)

    @property
    def numerical_expression(self):
        if self.prefactor is None:
            return sproduct(self.sign * self._operator, self.field.symbol)
        else:
            return sproduct(self.sign * self.prefactor, self._operator, self.field.symbol)


class ComposedOperatorsTerm(SingleArithmeticTerm):

    def __init__(self, field, operators, operators_args, inner_product_definition=None, prefactor=None, name='', sign=1):

        if len(operators_args) != len(operators):
            raise ValueError('Too many or too few operators arguments provided')
        SingleArithmeticTerm.__init__(self, field, inner_product_definition, prefactor, name, sign)
        self.prefactor = prefactor
        self._operators = list()
        for op, args in zip(operators, operators_args):
            if not isinstance(args, (tuple, list)):
                args = tuple([args])
            self._operators.append(op(*args))

    @property
    def symbolic_expression(self):
        expr = sproduct(*self._operators)
        expr = sproduct(expr, self.field.symbol)
        if self.prefactor is not None:
            expr = sproduct(self.prefactor.symbol, expr)
        return sproduct(self.sign, expr)

    @property
    def numerical_expression(self):
        expr = sproduct(*self._operators)
        expr = sproduct(expr, self.field.symbol)
        if self.prefactor is not None:
            expr = sproduct(self.prefactor, expr)
        return sproduct(self.sign, expr)
