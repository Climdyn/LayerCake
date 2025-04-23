
from layercake.arithmetic.terms.base import ArithmeticTerm
from layercake.utils.commutativity import disable_commutativity
from sympy import Mul


class LinearTerm(ArithmeticTerm):

    def __init__(self, field, inner_product_definition, parameter, name=''):

        ArithmeticTerm.__init__(self, field, inner_product_definition, name)
        self.parameter = parameter

    @property
    def symbolic_expression(self):
        return Mul(self.parameter.symbol, self.field.symbol, evaluate=False)

    @property
    def numerical_expression(self):
        return Mul(self.parameter, self.field.symbol, evaluate=False)

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
