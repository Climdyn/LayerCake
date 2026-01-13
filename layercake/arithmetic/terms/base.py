
"""

    Arithmetic terms definition module
    ==================================

    This module defines base classes for partial differential equation arithmetic terms.
    The corresponding objects hold the symbolic representation of the terms and their decomposition
    on given function basis.

    Description of the classes
    --------------------------

    * :class:`ArithmeticTerms`: General base class for partial differential equation arithmetic terms.
    * :class:`SingleArithmeticTerm`: Base class for single arithmetic terms (singleton) involving the field over which the partial differential equation acts.
    * :class:`OperationOnTerms`: Base class for operations on arithmetic terms. Perform the same operation on multiple terms.

"""

from abc import ABC, abstractmethod
import sparse as sp
import os
from pebble import ProcessPool as PebblePool
from multiprocessing import cpu_count
from itertools import product
from copy import deepcopy

from sympy import ImmutableSparseMatrix, ImmutableSparseNDimArray, Lambda, symbols
from sympy.core.function import BadSignatureError
from layercake.arithmetic.symbolic.operators import evaluate_expr
from layercake.utils.commutativity import enable_commutativity, disable_commutativity
from layercake.inner_products.definition import InnerProductDefinition
from layercake.arithmetic.utils import sproduct
from layercake.utils.symbolic_tensor import remove_dic_zeros
from layercake.utils.parallel import parallel_integration, parallel_symbolic_evaluation

if 'LAYERCAKE_PARALLEL_METHOD' not in os.environ:
    from concurrent.futures import ThreadPoolExecutor as Pool
else:
    if os.environ['LAYERCAKE_PARALLEL_METHOD'] == 'processes':
        from concurrent.futures import ProcessPoolExecutor as Pool
    elif os.environ['LAYERCAKE_PARALLEL_METHOD'] == 'threads':
        from concurrent.futures import ThreadPoolExecutor as Pool
    else:
        from concurrent.futures import ThreadPoolExecutor as Pool


class ArithmeticTerms(ABC):
    """General base class for partial differential equation arithmetic term(s).
    Holds the symbolic representation of (possibly multiple) term(s) and his(their) decomposition(s) on a given function basis.

    More precisely, models a term :math:`\\pm \\, T(u_1, u_2)` in the partial differential equation, where
    :math:`u_1, u_2` are the coordinates of the model.
    Upon decomposition on function basis, it can be represented as a tensor :math:`\\mathcal{T}_{i_1, \\ldots, i_r}` where :math:`r`
    is the tensor (and term(s)) rank.

    Parameters
    ----------
    name: str, optional
        Name of the term(s). Must be defined in subclasses.
    sign: int, optional
        Sign in front of the term(s). Either +1 or -1.
        Default to +1.

    Attributes
    ----------
    name: str, optional
        Name of the term(s). Must be defined in subclasses.
    sign: int, optional
        Sign in front of the term(s). Either +1 or -1.
    inner_products: None or ~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.tensor.array.ImmutableSparseNDimArray or sparse.COO(float)
        The inner products tensor of the term(s).
        Set initially to `None` (not computed).
    inner_product_definition: InnerProductDefinition
        Object defining the integral representation of the inner product that is used to compute the term(s) representation on a given function basis.
    """
    def __init__(self, name='', sign=1):

        self.sign = sign
        self.name = name
        self.inner_products = None
        self._rank = None
        self.inner_product_definition = None

    @property
    def rank(self):
        """int: Rank of the tensor storing the term(s) decomposition on the provided function basis."""
        if self.inner_products is not None:
            return self.inner_products.shape.__len__()
        else:
            return self._rank

    @property
    @abstractmethod
    def terms(self):
        """list(ArithmeticTerms): List of the terms."""
        pass

    @property
    @abstractmethod
    def _symbolic_expressions_list(self):
        pass

    @property
    @abstractmethod
    def _numerical_expressions_list(self):
        pass

    @property
    @abstractmethod
    def _symbolic_functions_list(self):
        pass

    @property
    @abstractmethod
    def _numerical_functions_list(self):
        pass

    @property
    @abstractmethod
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: The symbolic expression of the term(s). Only contains symbols."""
        pass

    @property
    @abstractmethod
    def numerical_expression(self):
        """~sympy.core.expr.Expr: The numeric expression of the term(s), with parameters replaced by their numerical value."""
        pass

    @property
    @abstractmethod
    def symbolic_function(self):
        """~sympy.core.expr.Expr: The symbolic expression of the term(s), but as a symbolic functional. Only contains symbols."""
        pass

    @property
    @abstractmethod
    def numerical_function(self):
        """~sympy.core.expr.Expr: The numeric expression of the term(s), as a symbolic functional,
        but with parameters replaced by their numerical value."""
        pass

    @staticmethod
    def _evaluate(func):
        return enable_commutativity(evaluate_expr(func))

    def _integrations(self, *basis, inner_product=None, numerical=False, parallelize=False, num_threads=None):
        """Returns the list of all the integrations to be computed to get the full tensor of inner products related to the term(s).
        Elements of the list includes indices locating the inner products in the tensor, inner product Sympy expression, and inner products integral arguments."""

        if len(basis) == 1:
            nmod = len(basis[0])
            nmodr = (range(nmod),) * self._rank
        else:
            if len(basis) != self._rank:
                raise ValueError('The number of basis provided should match the rank of the term.')
            nmod = tuple(map(lambda x: len(x), basis))
            nmodr = list()
            for n in nmod:
                nmodr.append(range(n))

        indices_list = [indices for indices in product(*nmodr)]
        if parallelize:
            if num_threads is None:
                num_threads = cpu_count()
            # with Pool(max_workers=num_threads) as pool:
            with Pool(max_workers=num_threads) as pool:
                args_list = parallel_symbolic_evaluation(pool, indices_list, inner_product, basis, numerical, self)
        else:
            args_list = [(indices, inner_product, self._inner_product_arguments(basis, indices, numerical=numerical))
                         for indices in product(*nmodr)]

        return args_list

    @abstractmethod
    def _inner_product_arguments(self, basis, indices, numerical=False):
        """Returns the tuple of all the arguments of the inner products integral for a given element of the tensor of
        inner products related to the term(s).
        To be implemented in subclasses.

        Parameters
        ----------
        basis: SymbolicBasis
            Symbolic function basis on which the term(s) must be decomposed.
        indices: list(int)
            Indices of the element in tensor for which to return the arguments of the inner products integrals.
        numerical: bool, optional
            Whether to provide numerical (with parameters' symbols replaced by their values)
            or symbolic basis function expressions in the output.
            Default to `False` (symbolic basis function expressions).

        Returns
        -------
        tuple(~sympy.core.expr.Expr)
            The tuple containing the arguments of the inner products integrals.
        """
        pass

    @abstractmethod
    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        """Compute the inner products tensor :math:`\\mathcal{T}_{i_1, \\ldots, i_r}`, either symbolic or numerical ones,
        representing the term(s) decomposed on a given function basis.
        Computations are parallelized on multiple CPUs.
        Results are stored in the :attr:`~ArithmeticTerms.inner_products` attribute.

        Parameters
        ----------
        basis: SymbolicBasis or list(SymbolicBasis)
            Symbolic basis function or list of symbolic function basis on which each element of the term(s) inner products must be decomposed.
        numerical: bool, optional
            Whether to compute numerical or symbolic inner products.
            Default to `False` (symbolic inner products as output).
        timeout: int or bool or None, optional
            Control the switch from symbolic to numerical integration. By default, `parallel_integration` workers will try to integrate
            |Sympy| expressions symbolically, but a fallback to numerical integration can be enforced.
            The options are:

            * `None`: This is the "full-symbolic" mode. No timeout will be applied, and the switch to numerical integration will never happen.
              Can result in very long and improbable computation time.
            * `True`: This is the "full-numerical" mode. Symbolic computations do not occur, and the workers try directly to integrate
              numerically.
            * `False`: Same as `None`.
            * An integer: defines a timeout after which, if a symbolic integration have not completed, the worker switch to the
              numerical integration.
        num_threads: None or int, optional
            Number of CPUs to use in parallel for the computations. If `None`, use all the CPUs available.
            Default to `None`.
        permute: bool, optional
            If `True`, applies all the possible permutations to the tensor indices
            from 1 to the rank of the tensor.
            Default to `False`, i.e. no permutation is applied.
        """
        pass

    def _compute_inner_products(self, *basis, numerical=False, timeout=None, num_threads=None, permute=False):
        """Internal function to handle the computation of the inner products, used by any subclasses.

        Parameters
        ----------
        *basis: SymbolicBasis
            Symbolic function basis on which each element of the term's inner products must be decomposed.
        numerical: bool, optional
            Whether to compute numerical (with parameters replaced with their values) or symbolic inner products.
            Default to `False` (symbolic inner products as output).
        timeout: int or bool or None, optional
            Control the switch from symbolic to numerical integration. By default, `parallel_integration` workers will try to integrate
            |Sympy| expressions symbolically, but a fallback to numerical integration can be enforced.
            The options are:

            * `None`: This is the "full-symbolic" mode. No timeout will be applied, and the switch to numerical integration will never happen.
              Can result in very long and improbable computation time.
            * `True`: This is the "full-numerical" mode. Symbolic computations do not occur, and the workers try directly to integrate
              numerically.
            * `False`: Same as `None`.
            * An integer: defines a timeout after which, if a symbolic integration have not completed, the worker switch to the
              numerical integration.
        num_threads: None or int, optional
            Number of CPUs to use in parallel for the computations. If `None`, use all the CPUs available.
            Default to `None`.
        permute: bool, optional
            If `True`, applies all the possible permutations to the tensor indices
            from 1 to the rank of the tensor.
            Default to `False`, i.e. no permutation is applied.
        """

        if num_threads is None:
            num_threads = cpu_count()

        parallelize = True
        if 'LAYERCAKE_PARALLEL_METHOD' in os.environ:
            if os.environ['LAYERCAKE_PARALLEL_METHOD'] == 'none':
                parallelize = False

        args_list = self._integrations(*basis,
                                       inner_product=self.inner_product_definition.inner_product,
                                       numerical=numerical, parallelize=parallelize,
                                       num_threads=num_threads)
        if len(basis) == 1:
            basis = basis[0]
            nmod = len(basis)
            rank = len(args_list[0][0])
            matrix_shape = (nmod,) * rank
            substitutions = basis.substitutions
        else:
            if len(basis) != self._rank:
                raise ValueError('The number of basis provided should match the rank of the term.')
            matrix_shape = tuple(map(lambda x: len(x), basis))
            substitutions = list()
            for b in basis:
                substitutions = substitutions + b.substitutions

        if numerical:
            res = sp.zeros(matrix_shape, dtype=float, format='dok')
        else:
            res = None
        with PebblePool(max_workers=num_threads) as pool:
            output = parallel_integration(pool, args_list, substitutions, res, timeout,
                                          symbolic_int=not numerical, permute=permute)
        if not numerical:
            output = remove_dic_zeros(output)
            if self._rank > 2:
                self.inner_products = ImmutableSparseNDimArray(output, matrix_shape)
            elif self._rank in [1, 2]:
                self.inner_products = ImmutableSparseMatrix(*matrix_shape, output)
            else:
                raise ValueError('Rank of the arithmetic term is wrong, something odd is happening.')
        else:
            self.inner_products = res.to_coo()

    @property
    @abstractmethod
    def latex(self):
        """str: Return a LaTeX representation of the term(s)."""
        return None

    def __repr__(self):
        return self.symbolic_expression.__str__()

    def __str__(self):
        return self.__repr__()

    def copy(self):
        """ArithmeticTerms: Return a copy of the term(s) object."""
        return deepcopy(self)

    def __neg__(self):
        neg = self.copy()
        neg.sign *= -1
        return neg


class SingleArithmeticTerm(ArithmeticTerms):
    """Base class for single arithmetic terms (singleton) involving the field over which the partial differential equation acts.
    Holds the symbolic representation of the term and his decomposition on given function basis.

    More precisely, models a term in the partial differential equation as a linear functional :math:`\\pm \\, T[\\psi](u_1, u_2)`,
    where :math:`\\psi` is the field solution of the equation, and the :math:`u_1, u_2` are the coordinates of the model.
    Upon decomposition on function basis, it can be represented as a tensor

    .. math:: \\mathcal{T}_{i_1, i_2} = \\left\\langle \\phi_{i_1} , \\pm \\, T[\\eta_{i_2}] \\right\\rangle

    where the :math:`\\phi_i`'s are basis functions provided by the user, and the :math:`\\eta_i`'s are basis functions on which
    the field :math:`\\psi` is decomposed. :math:`\\langle \\, , \\rangle` is the inner provided by the user.
    The rank :math:`r` of this kind of term (and its tensor rank) is thus always 2.

    Parameters
    ----------
    field: ~field.Field or ~field.ParameterField
        A field appearing in the partial differential equation.
    inner_product_definition: InnerProductDefinition, optional
        Object defining the integral representation of the inner product that is used to compute the term representation on a given function basis.
        If not provided, it will use the inner product definition found in the `field` object.
        Default to using the inner product definition found in the `field` object.
    prefactor: ~parameter.Parameter or ~field.FunctionField, optional
        Prefactor in front of the single term.
        Must be specified as a model parameter or a function field.
    name: str, optional
        Name of the term. Must be defined in subclasses.
    sign: int, optional
        Sign in front of the term(s). Either +1 or -1.
        Default to +1.

    Attributes
    ----------
    field: ~field.Field or ~field.ParameterField
        The field appearing in the partial differential equation.
    name: str
        Name of the term. Must be defined in subclasses.
    sign: int
        Sign in front of the term. Either +1 or -1.
    inner_products: None or ~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.tensor.array.ImmutableSparseNDimArray or sparse.COO(float)
        The inner products tensor of the term.
        Set initially to `None` (not computed).
    inner_product_definition: InnerProductDefinition
        Object defining the integral representation of the inner product that is used to compute the term representation on a given function basis.
    prefactor: ~parameter.Parameter or ~field.FunctionField
        Prefactor in front of the single term.
    """
    def __init__(self, field, inner_product_definition=None, prefactor=None, name='', sign=1):

        ArithmeticTerms.__init__(self, name, sign)
        self._rank = 2  # usual value for this kind of terms, can be changed in subclasses if needed
        self.field = field
        self.prefactor = prefactor
        if inner_product_definition is not None:
            self.inner_product_definition = inner_product_definition
        else:
            self.inner_product_definition = field.inner_product_definition

    @property
    def terms(self):
        """list(ArithmeticTerms): List of the terms. Here a single term is returned in the list."""
        return [self]

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
        """~sympy.core.expr.Expr: The symbolic expression of the term(s), but as a symbolic functional. Only contains symbols."""
        foo = disable_commutativity(self.symbolic_expression)
        ss = disable_commutativity(self.field.symbol)
        try:
            return Lambda(ss, foo)
        except BadSignatureError:
            _x = symbols('_x')
            return Lambda(_x, foo)

    @property
    def numerical_function(self):
        """~sympy.core.expr.Expr: The numeric expression of the term(s), as a symbolic functional, but with parameters replaced by their numerical value."""
        foo = disable_commutativity(self.numerical_expression)
        ss = disable_commutativity(self.field.symbol)
        try:
            return Lambda(ss, foo)
        except BadSignatureError:
            _x = symbols('_x')
            return Lambda(_x, foo)

    def _inner_product_arguments(self, basis, indices, numerical=False):
        """Returns the tuple of all the arguments of the inner products integral for a given element of the tensor of
        inner products related to the term.

        Parameters
        ----------
        basis: list(SymbolicBasis)
            List of symbolic function basis on which each element of the argument tuple must be decomposed.
        indices: list(int)
            Indices of the element in tensor for which to return the arguments of the inner products integrals.
        numerical: bool, optional
            Whether to provide numerical (with parameters' symbols replaced by their values)
            or symbolic basis function expressions in the output.
            Default to `False` (symbolic basis function expressions).

        Returns
        -------
        tuple(~sympy.core.expr.Expr)
            The tuple containing the arguments of the inner products integrals.

        """
        if numerical:
            funcs_list = self._numerical_functions_list
        else:
            funcs_list = self._symbolic_functions_list
        res = [None, None]
        for i, k in enumerate(indices):
            if i == 0:
                if len(basis) > 1:
                    res[0] = basis[i][k]
                else:
                    res[0] = basis[0][k]
            else:
                if len(basis) > 1:
                    res[1] = self._evaluate(funcs_list[i-1](disable_commutativity(basis[i][k])))
                else:
                    res[1] = self._evaluate(funcs_list[i-1](disable_commutativity(basis[0][k])))

        return tuple(res)

    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        """Compute the inner products tensor, either symbolic or numerical ones, representing the term decomposed on a given function basis.
        Computations are parallelized on multiple CPUs.
        Results are stored in the :attr:`~SingleArithmeticTerm.inner_products` attribute.

        Parameters
        ----------
        basis: SymbolicBasis
            Symbolic function basis on which the term must be decomposed, i.e integrated with the term's field's function basis.
        numerical: bool, optional
            Whether to compute numerical or symbolic inner products.
            Default to `False` (symbolic inner products as output).
        timeout: int or bool or None, optional
            Control the switch from symbolic to numerical integration. By default, `parallel_integration` workers will try to integrate
            |Sympy| expressions symbolically, but a fallback to numerical integration can be enforced.
            The options are:

            * `None`: This is the "full-symbolic" mode. No timeout will be applied, and the switch to numerical integration will never happen.
              Can result in very long and improbable computation time.
            * `True`: This is the "full-numerical" mode. Symbolic computations do not occur, and the workers try directly to integrate
              numerically.
            * `False`: Same as `None`.
            * An integer: defines a timeout after which, if a symbolic integration have not completed, the worker switch to the
              numerical integration.
        num_threads: None or int, optional
            Number of CPUs to use in parallel for the computations. If `None`, use all the CPUs available.
            Default to `None`.
        permute: bool, optional
            If `True`, applies all the possible permutations to the tensor indices
            from 1 to the rank of the tensor.
            Default to `False`, i.e. no permutation is applied.
        """
        basis_list = (basis, self.field.basis)
        self._compute_inner_products(*basis_list, numerical=numerical, timeout=timeout, num_threads=num_threads, permute=permute)


class OperationOnTerms(ArithmeticTerms):
    """Base class for operations on arithmetic terms. Perform the same operation on multiple terms.
    Holds the symbolic representation of the result and his decomposition on a given function basis.

    More precisely, models a term in the partial differential equation as a provided operation noted :math:`\\wedge`, acting
    on multiple multilinear functional terms :math:`\\pm \\, T(u_1, u_2) = \\bigwedge_{i=1}^k T_i[\\psi^i_1, \\ldots, \\psi^i_{j_i}] (u_1, u_2)`,
    where the :math:`\\psi^i_k`'s are the :math:`j_i` fields (possibly the same) on which the functional :math:`T_i` are acting,
    and the :math:`u_1, u_2` are the coordinates of the model.
    Upon decomposition on function basis, it can be represented as a tensor

    .. math:: \\mathcal{T}_{j, k_{1,1}, \\ldots, k_{1,j_1}, \\ldots, k_{l,1}, \\ldots, k_{l,{j_l}}} = \\left\\langle \\phi_{j} , \\pm \\, \\bigwedge_{i=1}^l T_i\\left[\\left(\\eta^i_1\\right)_{k_{i,1}}, \\ldots, \\left(\\eta^i_{j_i}\\right)_{k_{i,j_i}}\\right] \\right\\rangle

    where the :math:`\\phi_j`'s are basis functions provided by the user, and the :math:`\\left(\\eta^i_j\\right)_k`'s are basis functions on which
    the fields :math:`\\psi^i_j`'s are decomposed. :math:`\\langle \\, , \\rangle` is the inner provided by the user.
    The rank :math:`r` of this kind of term (and its tensor rank) is thus :math:`1+\\sum_{i=1}^l j_i`.

    Parameters
    ----------
    *terms: ArithmeticTerms
        The terms over which the operation is applied.
    **continuation_kwargs:
        Additional arguments passed to the object, see list below:

    Other Parameters
    ----------------
    inner_product_definition: InnerProductDefinition, optional
        Object defining the integral representation of the inner product that is used to compute the term representation on a given function basis.
        If not provided, it will use the inner product definition found in the `field` object.
        Default to using the inner product definition found in the `field` object.
    name: str, optional
        Name of the term(s). Must be defined in subclasses.
    sign: int, optional
        Sign in front of the term(s). Either +1 or -1.
        Default to +1.
    rank: int, optional
        Can be used to force the rank of the term, i.e. force the rank of the tensor storing the term(s) decomposition on the provided function basis.
        Compute the rank automatically if not provided.

    Attributes
    ----------
    name: str
        Name of the term. Must be defined in subclasses.
    sign: int
        Sign in front of the term. Either +1 or -1.
    inner_products: None or ~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.tensor.array.ImmutableSparseNDimArray or sparse.COO(float)
        The inner products tensor of the term.
        Set initially to `None` (not computed).
    inner_product_definition: InnerProductDefinition
        Object defining the integral representation of the inner product that is used to compute the term representation on a given function basis.
    """
    def __init__(self, *terms, **kwargs):

        if len(terms) < 2:
            raise ValueError('More than one term must be provided to this class.')

        if 'sign' in kwargs:
            sign = kwargs['sign']
        else:
            sign = 1

        for term in terms:
            if term.rank == 1:
                raise ValueError(f'The term {term} is of rank 1, which is not accepted in OperationOnTerms input.')

        ArithmeticTerms.__init__(self, sign=sign)
        if 'name' in kwargs:
            self.name = kwargs['name']
        compute_rank = False
        if 'rank' in kwargs:
            rank = kwargs['rank']
            if rank is not None:
                self._rank = rank
            else:
                compute_rank = True
        else:
            compute_rank = True

        self._terms = terms
        if compute_rank:
            self._compute_rank()

        self.inner_products = None
        if 'inner_product_definition' in kwargs:
            ipdef = kwargs['inner_product_definition']
            if issubclass(ipdef.__class__, InnerProductDefinition):
                self.inner_product_definition = ipdef
            elif isinstance(ipdef, int):
                self.inner_product_definition = terms[ipdef].inner_product_definition
            else:
                self.inner_product_definition = terms[0].inner_product_definition
        else:
            self.inner_product_definition = terms[0].inner_product_definition

    @property
    def terms(self):
        """list(ArithmeticTerms): List of the terms on which the operation acts."""
        return self._terms

    @abstractmethod
    def _compute_rank(self):
        """Routine to compute the rank of the inner products tensor, if not enforced by the user.

        Returns
        -------
        int
            The rank of the inner products tensor.
        """
        pass

    @property
    def number_of_terms(self):
        """int: Number of the terms on which the operation acts."""
        return self._terms.__len__()

    @abstractmethod
    def operation(self, *terms, evaluate=False):
        """Operation acting on the terms. Must be defined in subclasses.

        Parameters
        ----------
        *terms: ArithmeticTerms
            Terms on which the operation must be applied.
        evaluate: bool
            Whether to let |Sympy| evaluate the operation or not.
            Default to `False`.

        Returns
        -------
        ~sympy.core.expr.Expr
            The result of the operation on the terms, as a |Sympy| symbolic expression.

        """
        pass

    @property
    def symbolic_expression(self):
        """~sympy.core.expr.Expr: The symbolic expression of the result of the operation on the terms. Only contains symbols."""
        return sproduct(self.sign, self.operation(*self._symbolic_expressions_list))

    @property
    def numerical_expression(self):
        """~sympy.core.expr.Expr: The numerical expression of the result of the operation on the terms, with parameters replaced by their numerical value."""
        return sproduct(self.sign, self.operation(*self._numerical_expressions_list))

    @property
    def _symbolic_expressions_list(self):
        return list(map(lambda t: t.symbolic_expression, self._terms))

    @property
    def _numerical_expressions_list(self):
        return list(map(lambda t: t.numerical_expression, self._terms))

    @property
    def _symbolic_functions_list(self):
        return list(map(lambda t: t.symbolic_function, self._terms))

    @property
    def _numerical_functions_list(self):
        return list(map(lambda t: t.numerical_function, self._terms))

    @property
    def _fields_list(self):
        return list(map(lambda t: t.field, self._terms))

    @property
    def symbolic_function_dummy(self):
        """~sympy.core.expr.Expr: The symbolic expression of the result of the operation on the terms, but as a symbolic
        functional and with dummy symbols. Only contains symbols."""
        ss = symbols(" ".join(['x'+str(i) for i in range(self.number_of_terms)]))
        ssdc = list()
        for s in ss:
            ssdc.append(disable_commutativity(s))
        for i, ts in enumerate(self._terms):
            dcexpr = disable_commutativity(ts.symbolic_expression.replace(ts.field.symbol, ssdc[i]))
            if i == 0:
                foo = dcexpr
            else:
                foo = self.operation(foo, dcexpr)
        return Lambda(tuple(ssdc), sproduct(self.sign, foo))

    @property
    def symbolic_function(self):
        """~sympy.core.expr.Expr: The symbolic expression of the result of the operation on the terms, but as a symbolic functional.
        Only contains symbols."""
        ss = self._fields_list
        ssdc = list()
        for s in ss:
            sdc = disable_commutativity(s.symbol)
            if sdc not in ssdc:
                ssdc.append(sdc)
        for i, ts in enumerate(self._terms):
            dcexpr = disable_commutativity(ts.symbolic_expression)
            if i == 0:
                foo = dcexpr
            else:
                foo = self.operation(foo, dcexpr)
        return Lambda(tuple(ssdc), sproduct(self.sign, foo))

    @property
    def numerical_function_dummy(self):
        """~sympy.core.expr.Expr: The numerical expression of the result of the operation on the terms,
        as a symbolic functional with dummy symbols, and with parameters replaced by their numerical value."""
        ss = symbols(" ".join(['x'+str(i) for i in range(self.number_of_terms)]))
        ssdc = list()
        for s in ss:
            ssdc.append(disable_commutativity(s))
        for i, ts in enumerate(self._terms):
            dcexpr = disable_commutativity(ts.numerical_expression.replace(ts.field.symbol, ssdc[i]))
            if i == 0:
                foo = dcexpr
            else:
                foo = self.operation(foo, dcexpr)
        return Lambda(tuple(ssdc), sproduct(self.sign, foo))

    @property
    def numerical_function(self):
        """~sympy.core.expr.Expr: The numerical expression of the result of the operation on the terms, as a symbolic functional,
        and with parameters replaced by their numerical value."""
        ss = self._fields_list
        ssdc = list()
        for s in ss:
            sdc = disable_commutativity(s.symbol)
            if sdc not in ssdc:
                ssdc.append(sdc)
        for i, ts in enumerate(self._terms):
            dcexpr = disable_commutativity(ts.numerical_expression)
            if i == 0:
                foo = dcexpr
            else:
                foo = self.operation(foo, dcexpr)
        return Lambda(tuple(ssdc), sproduct(self.sign, foo))

    def _inner_product_arguments(self, basis, indices, numerical=False):
        """Returns the tuple of all the arguments of the inner products integral for a given element of the tensor of
        inner products related to operation on the terms.

        Parameters
        ----------
        basis: list(SymbolicBasis)
            List of symbolic function basis on which each element of the argument tuple must be decomposed.
        indices: list(int)
            Indices of the element in tensor for which to return the arguments of the inner products integrals.
        numerical: bool, optional
            Whether to provide numerical (with parameters' symbols replaced by their values)
            or symbolic basis function expressions in the output.
            Default to `False` (symbolic basis function expressions).

        Returns
        -------
        tuple(~sympy.core.expr.Expr)
            The tuple containing the arguments of the inner products integrals.

        """
        if numerical:
            funcs_list = self._numerical_functions_list
        else:
            funcs_list = self._symbolic_functions_list
        res = [None, None]
        for i, k in enumerate(indices):
            if i == 0:
                if len(basis) > 1:
                    res[0] = basis[i][k]
                else:
                    res[0] = basis[0][k]
            else:
                if i == 1:
                    if len(basis) > 1:
                        res[1] = self._evaluate(funcs_list[i-1](disable_commutativity(basis[i][k])))
                    else:
                        res[1] = self._evaluate(funcs_list[i-1](disable_commutativity(basis[0][k])))
                else:
                    if len(basis) > 1:
                        res[1] = self.operation(res[1], self._evaluate(funcs_list[i-1](disable_commutativity(basis[i][k]))))
                    else:
                        res[1] = self.operation(res[1], self._evaluate(funcs_list[i-1](disable_commutativity(basis[0][k]))))

        return tuple(res)

    @abstractmethod
    def _create_inner_products_basis_list(self, basis):
        """Function defining the list of symbolic function basis specified in order to compute the inner products
        for the operation on terms. Must be defined in subclasses.

        Parameters
        ----------
        basis: SymbolicBasis
            Basis to put on the left-hand side of the inner products.

        Returns
        -------
        tuple(SymbolicBasis)
            List of symbolic function basis used to compute the inner products.

        """
        pass

    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        """Compute the inner products tensor, either symbolic or numerical ones, representing the term(s) decomposed on a given function basis.
        Computations are parallelized on multiple CPUs.
        Results are stored in the :attr:`~OperationOnTerms.inner_products` attribute.

        Parameters
        ----------
        basis: list(SymbolicBasis)
            List of symbolic function basis on which each term of the operation on the terms must be decomposed.
        numerical: bool, optional
            Whether to compute numerical or symbolic inner products.
            Default to `False` (symbolic inner products as output).
        timeout: int or bool or None, optional
            Control the switch from symbolic to numerical integration. By default, `parallel_integration` workers will try to integrate
            |Sympy| expressions symbolically, but a fallback to numerical integration can be enforced.
            The options are:

            * `None`: This is the "full-symbolic" mode. No timeout will be applied, and the switch to numerical integration will never happen.
              Can result in very long and improbable computation time.
            * `True`: This is the "full-numerical" mode. Symbolic computations do not occur, and the workers try directly to integrate
              numerically.
            * `False`: Same as `None`.
            * An integer: defines a timeout after which, if a symbolic integration have not completed, the worker switch to the
              numerical integration.
        num_threads: None or int, optional
            Number of CPUs to use in parallel for the computations. If `None`, use all the CPUs available.
            Default to `None`.
        permute: bool, optional
            If `True`, applies all the possible permutations to the tensor indices
            from 1 to the rank of the tensor.
            Default to `False`, i.e. no permutation is applied.
        """
        basis_list = self._create_inner_products_basis_list(basis)
        self._compute_inner_products(*basis_list, numerical=numerical, timeout=timeout, num_threads=num_threads, permute=permute)
        self.inner_products = self.sign * self.inner_products
