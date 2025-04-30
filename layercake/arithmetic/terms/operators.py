
from layercake.arithmetic.terms.base import SingleArithmeticTerm
from layercake.utils.commutativity import disable_commutativity
from sympy import Mul


class OperatorTerm(SingleArithmeticTerm):

    def __init__(self, field, inner_product_definition, operator, operator_args, parameter=None, name=''):

        SingleArithmeticTerm.__init__(self, field, inner_product_definition, name)
        self.parameter = parameter
        if not isinstance(operator_args, (tuple, list)):
            operator_args = tuple([operator_args])
        self._operator = operator(*operator_args)

    @property
    def symbolic_expression(self):
        if self.parameter is None:
            return Mul(self._operator, self.field.symbol, evaluate=False)
        else:
            return Mul(self.parameter.symbol, Mul(self._operator, self.field.symbol, evaluate=False), evaluate=False)

    @property
    def numerical_expression(self):
        if self.parameter is None:
            return Mul(self._operator, self.field.symbol, evaluate=False)
        else:
            return Mul(self.parameter, Mul(self._operator, self.field.symbol, evaluate=False), evaluate=False)


class ComposedOperatorsTerm(SingleArithmeticTerm):

    def __init__(self, field, inner_product_definition, operators, operators_args, parameter=None, name=''):

        if len(operators_args) != len(operators):
            raise ValueError('Too many or too few operators arguments provided')
        SingleArithmeticTerm.__init__(self, field, inner_product_definition, name)
        self.parameter = parameter
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
        if self.parameter is not None:
            expr = Mul(self.parameter.symbol, expr, evaluate=False)
        return expr

    @property
    def numerical_expression(self):
        expr = self._operators[0]
        for op in self._operators[1:]:
            expr = Mul(expr, op, evaluate=False)
        expr = Mul(expr, self.field.symbol, evaluate=False)
        if self.parameter is not None:
            expr = Mul(self.parameter, expr, evaluate=False)
        return expr

