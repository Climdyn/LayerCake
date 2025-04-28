
from layercake.arithmetic.terms.base import ArithmeticTerm
from layercake.utils.commutativity import disable_commutativity
from layercake.utils.operators import Laplacian
from sympy import Mul


class LaplacianTerm(ArithmeticTerm):

    def __init__(self, field, inner_product_definition, parameter=None, name=''):

        ArithmeticTerm.__init__(self, field, inner_product_definition, name)
        self._rank = 1
        self.parameter = parameter
        self._operator = Laplacian(field.coordinate_system)

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
