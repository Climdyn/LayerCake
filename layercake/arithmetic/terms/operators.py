
from layercake.arithmetic.terms.base import SingleArithmeticTerm
from sympy import Mul


class OperatorTerm(SingleArithmeticTerm):

    def __init__(self, field, operator, operator_args, inner_product_definition=None, prefactor=None, name=''):

        SingleArithmeticTerm.__init__(self, field, inner_product_definition, prefactor, name)
        if not isinstance(operator_args, (tuple, list)):
            operator_args = tuple([operator_args])
        self._operator = operator(*operator_args)

    @property
    def symbolic_expression(self):
        if self.prefactor is None:
            return Mul(self._operator, self.field.symbol, evaluate=False)
        else:
            return Mul(self.prefactor.symbol, Mul(self._operator, self.field.symbol, evaluate=False), evaluate=False)

    @property
    def numerical_expression(self):
        if self.prefactor is None:
            return Mul(self._operator, self.field.symbol, evaluate=False)
        else:
            return Mul(self.prefactor, Mul(self._operator, self.field.symbol, evaluate=False), evaluate=False)


class ComposedOperatorsTerm(SingleArithmeticTerm):

    def __init__(self, field, operators, operators_args, inner_product_definition=None, prefactor=None, name=''):

        if len(operators_args) != len(operators):
            raise ValueError('Too many or too few operators arguments provided')
        SingleArithmeticTerm.__init__(self, field, inner_product_definition, prefactor, name)
        self.prefactor = prefactor
        self._operators = list()
        for op, args in zip(operators, operators_args):
            if not isinstance(args, (tuple, list)):
                args = tuple([args])
            self._operators.append(op(*args))

    @property
    def symbolic_expression(self):
        expr = self._operators[0]
        for op in self._operators[1:]:
            expr = Mul(expr, op, evaluate=False)
        expr = Mul(expr, self.field.symbol, evaluate=False)
        if self.prefactor is not None:
            expr = Mul(self.prefactor.symbol, expr, evaluate=False)
        return expr

    @property
    def numerical_expression(self):
        expr = self._operators[0]
        for op in self._operators[1:]:
            expr = Mul(expr, op, evaluate=False)
        expr = Mul(expr, self.field.symbol, evaluate=False)
        if self.prefactor is not None:
            expr = Mul(self.prefactor, expr, evaluate=False)
        return expr

