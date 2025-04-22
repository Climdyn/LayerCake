
from layercake.arithmetic.terms.base import ArithmeticTerm
from layercake.utils.d import D, evaluate_expr
from layercake.utils.commutativity import enable_commutativity, disable_commutativity
from sympy import Lambda, Mul


class DirectionalDerivativeTerm(ArithmeticTerm):

    def __init__(self, field, inner_product_definition, direction, parameter):

        ArithmeticTerm.__init__(self, field, inner_product_definition)
        self.name = 'Directional derivative term'
        self.parameter = parameter
        self.direction = direction
        self._operator = D(direction)

    @property
    def symbolic_expression(self):
        return Mul(self.parameter.symbol, Mul(self._operator, self.field.symbol, evaluate=False), evaluate=False)

    @property
    def numerical_expression(self):
        return Mul(self.parameter, Mul(self._operator, self.field.symbol, evaluate=False), evaluate=False)

    @property
    def symbolic_function(self):
        foo = disable_commutativity(self.symbolic_expression)
        ss = foo.args[-1]
        return Lambda(ss, foo)

    @property
    def numerical_function(self):
        foo = disable_commutativity(self.numerical_expression)
        ss = foo.args[-1]
        return Lambda(ss, foo)

    @staticmethod
    def _evaluate(func):
        return enable_commutativity(evaluate_expr(func))

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
