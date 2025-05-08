
from layercake.arithmetic.terms.base import ArithmeticTerm
from layercake.variables.field import ParameterField

from layercake.utils.commutativity import disable_commutativity
from layercake.arithmetic.utils import sproduct
from sympy import Lambda


class ConstantTerm(ArithmeticTerm):

    def __init__(self, parameters_field, name='', sign=1):

        if not isinstance(parameters_field, ParameterField):
            raise ValueError('Input field for constant field must be a ParameterField object.')

        ArithmeticTerm.__init__(self, name, sign=sign)

        self._rank = 1
        self.field = parameters_field

    @property
    def _symbolic_expressions_list(self):
        return [self.symbolic_expression]

    @property
    def _numerical_expressions_list(self):
        return [self.numerical_expression]

    @property
    def _symbolic_functions_list(self):
        return [self.symbolic_function]

    @property
    def _numerical_functions_list(self):
        return [self.numerical_function]

    @property
    def symbolic_function(self):
        foo = disable_commutativity(self.symbolic_expression)
        ss = disable_commutativity(self.field.symbol)
        return Lambda(ss, foo)

    @property
    def numerical_function(self):
        foo = disable_commutativity(self.numerical_expression)
        ss = disable_commutativity(self.field.symbol)
        return Lambda(ss, foo)

    @property
    def symbolic_expression(self):
        return sproduct(self.sign, self.field.symbol)

    @property
    def numerical_expression(self):
        return sproduct(self.sign, self.symbolic_expression)

    def _inner_product_arguments(self, basis, indices, numerical=False):
        pass

    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        pass
