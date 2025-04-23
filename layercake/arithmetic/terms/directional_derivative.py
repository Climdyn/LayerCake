
from layercake.arithmetic.terms.base import ArithmeticTerm
from layercake.utils.operators import D
from layercake.utils.commutativity import disable_commutativity
from layercake.variables.coordinate import Coordinate
from sympy import Mul, S


class DirectionalDerivativeTerm(ArithmeticTerm):

    def __init__(self, field, inner_product_definition, direction, parameter, name='', infinitesimal_length=None):

        ArithmeticTerm.__init__(self, field, inner_product_definition, name)
        self.parameter = parameter
        self.direction = direction
        if infinitesimal_length is not None:
            self._ds = 1 / infinitesimal_length
        elif isinstance(direction, Coordinate):
            self._ds = 1 / direction.infinitesimal_length
        else:
            self._ds = 1 / S.One

        try:
            self._operator = Mul(self._ds, D(direction.symbol), evaluate=False)
        except AttributeError:
            self._operator = Mul(self._ds, D(direction), evaluate=False)


    @property
    def symbolic_expression(self):
        return Mul(self.parameter.symbol, Mul(self._operator, self.field.symbol, evaluate=False), evaluate=False)

    @property
    def numerical_expression(self):
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
