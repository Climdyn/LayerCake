
from layercake.arithmetic.terms.base import SingleArithmeticTerm
from layercake.utils.commutativity import disable_commutativity
from sympy import Mul


class OperatorTerm(SingleArithmeticTerm):

    def __init__(self, field, inner_product_definition, operator, operator_args, parameter=None, name=''):

        SingleArithmeticTerm.__init__(self, field, inner_product_definition, name)
        self._rank = 1
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

    def __init__(self, field, inner_product_definition, operator1, operator_args1,
                 operator2, operator_args2, parameter=None, name=''):

        SingleArithmeticTerm.__init__(self, field, inner_product_definition, name)
        self._rank = 1
        self.parameter = parameter
        if not isinstance(operator_args1, (tuple, list)):
            operator_args1 = tuple([operator_args1])
        self._operator1 = operator1(*operator_args1)
        if not isinstance(operator_args2, (tuple, list)):
            operator_args2 = tuple([operator_args2])
        self._operator2 = operator2(*operator_args2)

    @property
    def symbolic_expression(self):
        if self.parameter is None:
            return Mul(self._operator1,
                       Mul(self._operator2, self.field.symbol, evaluate=False), evaluate=False)
        else:
            return Mul(self.parameter.symbol,
                       Mul(self._operator1,
                           Mul(self._operator2, self.field.symbol, evaluate=False), evaluate=False),
                       evaluate=False)

    @property
    def numerical_expression(self):
        if self.parameter is None:
            return Mul(self._operator1,
                       Mul(self._operator2, self.field.symbol, evaluate=False), evaluate=False)
        else:
            return Mul(self.parameter,
                       Mul(self._operator1,
                           Mul(self._operator2, self.field.symbol, evaluate=False), evaluate=False),
                       evaluate=False)

