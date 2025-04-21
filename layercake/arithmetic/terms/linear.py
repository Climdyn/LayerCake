
from layercake.arithmetic.terms.base import ArithmeticTerm
from sympy import Lambda
import sparse as sp
from pebble import ProcessPool as Pool
from multiprocessing import cpu_count


class LinearTerm(ArithmeticTerm):

    def __init__(self, field, inner_product_definition, parameter):

        ArithmeticTerm.__init__(self, field, inner_product_definition)
        self.name = 'Linear term'
        self.parameter = parameter

    @property
    def symbolic_expression(self):
        return self.parameter.symbol * self.field.symbol

    @property
    def numerical_expression(self):
        return self.parameter * self.field.symbol

    @property
    def symbolic_function(self):
        return Lambda(self.field.symbol, self.symbolic_expression)

    @property
    def numerical_function(self):
        return Lambda(self.field.symbol, self.numerical_expression)

    def _integrations(self, basis, numerical=False):
        nmod = len(basis)
        if numerical:
            args_list = [[(i, j), self.inner_product_definition.inner_product,
                          (basis[i], self.numerical_function(basis[j]))]
                         for i in range(nmod)
                         for j in range(nmod)]
        else:
            args_list = [[(i, j), self.inner_product_definition.inner_product,
                          (basis[i], self.symbolic_function(basis[j]))]
                         for i in range(nmod)
                         for j in range(nmod)]

        return args_list

