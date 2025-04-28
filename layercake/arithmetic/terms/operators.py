
from layercake.arithmetic.terms.base import ArithmeticTerm
from layercake.utils.commutativity import disable_commutativity
from sympy import Mul


class OperatorTerm(ArithmeticTerm):

    def __init__(self, field, inner_product_definition, operator, operator_args, parameter=None, name=''):

        ArithmeticTerm.__init__(self, field, inner_product_definition, name)
        self._rank = 1
        self.parameter = parameter
        if isinstance(operator_args, list):
            operator_args = tuple(operator_args)
        elif not isinstance(operator_args, tuple):
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

    def _integrations(self, basis, numerical=False):
        nmod = len(basis)
        if numerical:
            args_list = [[(i, j), self.inner_product_definition.inner_product,
                          (basis[i], self._evaluate(self.numerical_function(disable_commutativity(basis[j]))))]
                         for i in range(nmod)
                         for j in range(nmod)]
        else:
            args_list = [[(i, j), self.inner_product_definition.inner_product,
                          (basis[i], self._evaluate(self.symbolic_function(disable_commutativity(basis[j]))))]
                         for i in range(nmod)
                         for j in range(nmod)]

        return args_list


class ComposedOperatorsTerm(ArithmeticTerm):

    def __init__(self, field, inner_product_definition, operator1, operator_args1,
                 operator2, operator_args2, parameter=None, name=''):

        ArithmeticTerm.__init__(self, field, inner_product_definition, name)
        self._rank = 1
        self.parameter = parameter
        if isinstance(operator_args1, list):
            operator_args1 = tuple(operator_args1)
        elif not isinstance(operator_args1, tuple):
            operator_args1 = tuple([operator_args1])
        self._operator1 = operator1(*operator_args1)
        if isinstance(operator_args2, list):
            operator_args2 = tuple(operator_args2)
        elif not isinstance(operator_args2, tuple):
            operator_args2 = tuple([operator_args2])
        self._operator2 = operator1(*operator_args2)

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

    def _integrations(self, basis, numerical=False):
        nmod = len(basis)
        if numerical:
            args_list = [[(i, j), self.inner_product_definition.inner_product,
                          (basis[i], self._evaluate(self.numerical_function(disable_commutativity(basis[j]))))]
                         for i in range(nmod)
                         for j in range(nmod)]
        else:
            args_list = [[(i, j), self.inner_product_definition.inner_product,
                          (basis[i], self._evaluate(self.symbolic_function(disable_commutativity(basis[j]))))]
                         for i in range(nmod)
                         for j in range(nmod)]

        return args_list
