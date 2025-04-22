
from layercake.arithmetic.terms.base import ArithmeticTerm
from layercake.utils.d import D, evaluateExpr
from sympy import Lambda, Expr


class DirectionalDerivativeTerm(ArithmeticTerm):

    def __init__(self, field, inner_product_definition, direction, parameter):

        ArithmeticTerm.__init__(self, field, inner_product_definition)
        self.name = 'Directional derivative term'
        self.parameter = parameter
        self.direction = direction
        self._operator = D(direction)

    @property
    def symbolic_expression(self):
        return self.parameter.symbol * self._operator * self.field.symbol

    @property
    def numerical_expression(self):
        return self.parameter * self._operator * self.field.symbol

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
                          (basis[i], evaluateExpr(self.numerical_function(basis[j])))]
                         for i in range(nmod)
                         for j in range(nmod)]
        else:
            args_list = [[(i, j), self.inner_product_definition.inner_product,
                          (basis[i], evaluateExpr(self.symbolic_function(basis[j])))]
                         for i in range(nmod)
                         for j in range(nmod)]

        return args_list
