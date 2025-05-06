import numpy as np

from layercake.arithmetic.terms.base import SingleArithmeticTerm
from layercake.variables.field import ParameterField


class ConstantTerm(SingleArithmeticTerm):

    def __init__(self, parameters_field, inner_product_definition=None, name=''):

        if not isinstance(parameters_field, ParameterField):
            raise ValueError('Input field for constant field must be a ParameterField object.')

        SingleArithmeticTerm.__init__(self, parameters_field, inner_product_definition, None, name)

        self._rank = 1

    @property
    def symbolic_expression(self):
        return self.field.symbol

    @property
    def numerical_expression(self):
        return self.symbolic_expression

    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        self._rank = 2
        basis_list = (basis, self.field.basis)
        self._compute_inner_products(*basis_list, numerical=numerical, timeout=timeout, num_threads=num_threads, permute=permute)
        self._rank = 1
        if numerical:
            self.inner_products = np.diagonal(self.inner_products)
        else:
            self.inner_products = self.inner_products.diagonal()
