
"""

    Operator arithmetic term definition module
    =================================================

    This module defines operator terms in partial differential equations.
    The corresponding objects hold the symbolic representation of the terms and their decomposition
    on given function basis.

    Description of the classes
    --------------------------

    * :class:`OperatorTerm`: Operator term in a partial differential equation, acting on fields of the equation.
    * :class:`ComposedOperatorsTerm`: Term representing the composition of multiple operators, acting on fields of the equation.

"""

from layercake.arithmetic.terms.base import SingleArithmeticTerm
from layercake.arithmetic.utils import sproduct


class OperatorTerm(SingleArithmeticTerm):
    """Operator term in a partial differential equation, acting on fields of the equation,
    and of the form :math:`\\pm \\, a \\, H \\psi(u_1, u_2)`, where :math:`H` is the operator,
    :math:`u_1, u_2` are the coordinates of the model, :math:`a` is a prefactor,
    and :math:`\\psi` is a field of the equation.

    Parameters
    ----------
    field: ~field.Field or ~field.ParameterField
        A field appearing in the partial differential equation, and on which the
        operator acts.
    operator: object
        Object or function returning the action of the operator on symbolic Sympy expressions.
        Must also have a `latex` attribute.
    operator_args: tuple
        Tuple of arguments to pass to the `operator` object or function.
    inner_product_definition: InnerProductDefinition, optional
        Object defining the integral representation of the inner product that is used
        to compute the term representation on a given function basis.
        If not provided, it will use the inner product definition found in the `field` object.
        Default to using the inner product definition found in the `field` object.
    prefactor: ~parameter.Parameter or ~field.FunctionField, optional
        Prefactor in front of the operator.
        Must be specified as a model parameter or a function field.
    name: str, optional
        Name of the term.
    sign: int, optional
        Sign in front of the term. Either +1 or -1.
        Default to +1.

    Attributes
    ----------
    field: ~field.Field or ~field.ParameterField
        The field appearing in the partial differential equation, and on which the
        operator acts.
    name: str
        Name of the term.
    sign: int
        Sign in front of the term. Either +1 or -1.
    inner_products: None or ~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.tensor.array.ImmutableSparseNDimArray or sparse.COO(float)
        The inner products tensor of the term.
        Set initially to `None` (not computed).
    inner_product_definition: InnerProductDefinition
        Object defining the integral representation of the inner product that is used to compute the term representation on a given function basis.
    prefactor: ~parameter.Parameter or ~field.FunctionField, optional
        Prefactor in front of the operator.
    """

    def __init__(self, field, operator, operator_args, inner_product_definition=None, prefactor=None, name='', sign=1):

        SingleArithmeticTerm.__init__(self, field, inner_product_definition, prefactor, name, sign=sign)
        if not isinstance(operator_args, (tuple, list)):
            operator_args = tuple([operator_args])
        self._operator = operator(*operator_args)

    @property
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: The symbolic expression of the operator. Only contains symbols."""
        if self.prefactor is None:
            return sproduct(self.sign * self._operator, self.field.symbol)
        else:
            return sproduct(self.sign * self.prefactor.symbol, self._operator, self.field.symbol)

    @property
    def numerical_expression(self):
        """~sympy.core.expr.Expr: The numeric expression of the operator, with parameters replaced by their numerical value."""
        if self.prefactor is None:
            return sproduct(self.sign * self._operator, self.field.symbol)
        else:
            if hasattr(self.prefactor, 'numerical_expression'):
                return sproduct(self.sign * self.prefactor.numerical_expression, self._operator, self.field.symbol)
            else:
                return sproduct(self.sign * self.prefactor, self._operator, self.field.symbol)

    @property
    def latex(self):
        """str: Return a LaTeX representation of the operator."""
        if self.sign > 0:
            s = f'+ '
        else:
            s = f'- '

        op = f'{self._operator.latex} {self.field.latex} '
        if self.prefactor is None:
            return s + op
        if hasattr(self.prefactor, 'latex'):
            if self.prefactor.latex is not None:
                s += f'{self.prefactor.latex} '
                return s + op
        if hasattr(self.prefactor, 'symbol'):
            if self.prefactor.symbol is not None:
                s += f'{self.prefactor.symbol} '
                return s + op

        s += f'{self.prefactor} '
        return s + op


class ComposedOperatorsTerm(SingleArithmeticTerm):
    """Term representing the composition :math:`\\circ` of multiple operators :math:`H_i`
    acting on fields of a partial differential equation, and of the form
    :math:`\\pm \\, a \\, H_1 \\circ H_2 \\ldots \\circ H_n \\psi(u_1, u_2)`, where :math:`u_1, u_2` are
    the coordinates of the model, :math:`a` is a prefactor, and :math:`\\psi` is a field of the equation.

    Parameters
    ----------
    field: ~field.Field or ~field.ParameterField
        A field appearing in the partial differential equation, and on which the composed
        operators act.
    operators: list(object)
        List of objects or functions returning the action of the operator on symbolic Sympy expressions.
        Each component of the list must also have a `latex` attribute.
    operators_args: list(tuple)
        Tuples of arguments to pass to the `operators` objects or functions, one tuple per operator.
    inner_product_definition: InnerProductDefinition, optional
        Object defining the integral representation of the inner product that is used
        to compute the term representation on a given function basis.
        If not provided, it will use the inner product definition found in the `field` object.
        Default to using the inner product definition found in the `field` object.
    prefactor: ~parameter.Parameter or ~field.FunctionField, optional
        Prefactor in front of the operator.
        Must be specified as a model parameter or a function field.
    name: str, optional
        Name of the term.
    sign: int, optional
        Sign in front of the term. Either +1 or -1.
        Default to +1.

    Attributes
    ----------
    field: ~field.Field or ~field.ParameterField
        The field appearing in the partial differential equation, and on which the
        operator acts.
    name: str
        Name of the term.
    sign: int
        Sign in front of the term. Either +1 or -1.
    inner_products: None or ~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.tensor.array.ImmutableSparseNDimArray or sparse.COO(float)
        The inner products tensor of the term.
        Set initially to `None` (not computed).
    inner_product_definition: InnerProductDefinition
        Object defining the integral representation of the inner product that is used to compute
        the term representation on a given function basis.
    prefactor: ~parameter.Parameter or ~field.FunctionField, optional
        Prefactor in front of the operator.
    """

    def __init__(self, field, operators, operators_args, inner_product_definition=None, prefactor=None, name='', sign=1):

        if len(operators_args) != len(operators):
            raise ValueError('Too many or too few operators arguments provided')
        SingleArithmeticTerm.__init__(self, field, inner_product_definition, prefactor, name, sign)
        self.prefactor = prefactor
        self._operators = list()
        for op, args in zip(operators, operators_args):
            if not isinstance(args, (tuple, list)):
                args = tuple([args])
            self._operators.append(op(*args))

    @property
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: The symbolic expression of the operators. Only contains symbols."""
        expr = sproduct(*self._operators)
        expr = sproduct(expr, self.field.symbol)
        if self.prefactor is not None:
            expr = sproduct(self.prefactor.symbol, expr)
        return sproduct(self.sign, expr)

    @property
    def numerical_expression(self):
        """~sympy.core.expr.Expr: The numeric expression of the operators, with parameters replaced by their numerical value."""
        expr = sproduct(*self._operators)
        expr = sproduct(expr, self.field.symbol)
        if self.prefactor is not None:
            if hasattr(self.prefactor, 'numerical_expression'):
                expr = sproduct(self.prefactor.numerical_expression, expr)
            else:
                expr = sproduct(self.prefactor, expr)
        return sproduct(self.sign, expr)

    @property
    def latex(self):
        """str: Return a LaTeX representation of the operators."""
        if self.sign > 0:
            s = f'+ '
        else:
            s = f'- '

        op = f'{self.field.latex} '
        for oper in self._operators[::-1]:
            op = f'{oper.latex} ' + op
        if self.prefactor is None:
            return s + op
        if hasattr(self.prefactor, 'latex'):
            if self.prefactor.latex is not None:
                s += f'{self.prefactor.latex} '
                return s + op
        if hasattr(self.prefactor, 'symbol'):
            if self.prefactor.symbol is not None:
                s += f'{self.prefactor.symbol} '
                return s + op

        s += f'{self.prefactor} '
        return s + op
