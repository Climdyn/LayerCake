
from abc import ABC, abstractmethod
import sparse as sp
from pebble import ProcessPool as Pool
from concurrent.futures import TimeoutError
from multiprocessing import cpu_count
from sympy.utilities.iterables import multiset_permutations
from itertools import product
from copy import deepcopy

from scipy.integrate import dblquad
from sympy import ImmutableSparseMatrix, ImmutableSparseNDimArray, lambdify, Lambda, symbols
from layercake.arithmetic.symbolic.operators import evaluate_expr
from layercake.utils.commutativity import enable_commutativity, disable_commutativity
from layercake.inner_products.definition import InnerProductDefinition
from layercake.arithmetic.utils import sproduct


class ArithmeticTerms(ABC):
    """Base class for arithmetic terms"""
    def __init__(self, name='', sign=1):

        self.sign = sign
        self.name = name
        self.inner_products = None
        self._rank = None
        self.inner_product_definition = None

    @property
    def rank(self):
        if self.inner_products is not None:
            return self.inner_products.shape.__len__()
        else:
            return self._rank

    @property
    @abstractmethod
    def terms(self):
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
        pass

    @property
    @abstractmethod
    def numerical_expression(self):
        pass

    @property
    @abstractmethod
    def symbolic_function(self):
        pass

    @property
    @abstractmethod
    def numerical_function(self):
        pass

    @staticmethod
    def _evaluate(func):
        return enable_commutativity(evaluate_expr(func))

    def _integrations(self, *basis, inner_product=None, numerical=False):
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
        args_list = [(indices, inner_product, self._inner_product_arguments(basis, indices, numerical=numerical))
                     for indices in product(*nmodr)]

        return args_list

    @abstractmethod
    def _inner_product_arguments(self, basis, indices, numerical=False):
        pass

    @abstractmethod
    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        pass

    def _compute_inner_products(self, *basis, numerical=False, timeout=None, num_threads=None, permute=False):

        if num_threads is None:
            num_threads = cpu_count()

        args_list = self._integrations(*basis,
                                       inner_product=self.inner_product_definition.inner_product,
                                       numerical=numerical)
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
        with Pool(max_workers=num_threads) as pool:
            output = _parallel_compute(pool, args_list, substitutions, res, timeout,
                                       symbolic_int=not numerical, permute=permute)
        if not numerical:
            if self._rank > 2:
                self.inner_products = ImmutableSparseNDimArray(output, matrix_shape)
            elif self._rank in [1, 2]:
                self.inner_products = ImmutableSparseMatrix(*matrix_shape, output)
            else:
                raise ValueError('Rank of the arithmetic term is wrong, something odd is happening.')
        else:
            self.inner_products = res.to_coo()

    def __repr__(self):
        return self.symbolic_expression.__str__()

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return deepcopy(self)

    def __neg__(self):
        neg = self.copy()
        neg.sign *= -1
        return neg


class SingleArithmeticTerm(ArithmeticTerms):
    """Base class for single arithmetic terms"""
    def __init__(self, field, inner_product_definition=None, prefactor=None, name='', sign=1):

        ArithmeticTerms.__init__(self, name, sign)
        self._rank = 2
        self.field = field
        self.prefactor = prefactor
        if inner_product_definition is not None:
            self.inner_product_definition = inner_product_definition
        else:
            self.inner_product_definition = field.inner_product_definition

    @property
    def terms(self):
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
        foo = disable_commutativity(self.symbolic_expression)
        ss = disable_commutativity(self.field.symbol)
        return Lambda(ss, foo)

    @property
    def numerical_function(self):
        foo = disable_commutativity(self.numerical_expression)
        ss = disable_commutativity(self.field.symbol)
        return Lambda(ss, foo)

    @staticmethod
    def _evaluate(func):
        return enable_commutativity(evaluate_expr(func))

    def _inner_product_arguments(self, basis, indices, numerical=False):
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
        basis_list = (basis, self.field.basis)
        self._compute_inner_products(*basis_list, numerical=numerical, timeout=timeout, num_threads=num_threads, permute=permute)


class OperationOnTerms(ArithmeticTerms):
    """Base class for operations on arithmetic terms"""
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
        return self._terms

    @abstractmethod
    def _compute_rank(self):
        pass

    @property
    def number_of_terms(self):
        return self._terms.__len__()

    @abstractmethod
    def operation(self, *terms, evaluate=False):
        pass

    @property
    def symbolic_expression(self):
        return sproduct(self.sign, self.operation(*self._symbolic_expressions_list))

    @property
    def numerical_expression(self):
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

    @staticmethod
    def _evaluate(func):
        return enable_commutativity(evaluate_expr(func))

    @property
    def symbolic_function_dummy(self):
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
        pass

    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):
        basis_list = self._create_inner_products_basis_list(basis)
        self._compute_inner_products(*basis_list, numerical=numerical, timeout=timeout, num_threads=num_threads, permute=permute)
        self.inner_products = self.sign * self.inner_products


def _apply(ls):
    return ls[0], ls[1](*ls[2])


def _num_apply(ls):
    integrand = ls[1](*ls[2], integrand=True)

    num_integrand = integrand[0].subs(ls[3])
    func = lambdify((integrand[1][0], integrand[2][0]), num_integrand, 'numpy')

    try:
        a = integrand[2][1].subs(ls[3])
    except:
        a = integrand[2][1]
    try:
        a = a.evalf()
    except:
        pass
    try:
        b = integrand[2][2].subs(ls[3])
    except:
        b = integrand[2][2]
    try:
        b = b.evalf()
    except:
        pass
    try:
        gfun = integrand[1][1].subs(ls[3])
    except:
        gfun = integrand[1][1]
    try:
        gfun = gfun.evalf()
    except:
        pass
    try:
        hfun = integrand[1][2].subs(ls[3])
    except:
        hfun = integrand[1][2]
    try:
        hfun = hfun.evalf()
    except:
        pass

    res = dblquad(func, a, b, gfun, hfun)

    if abs(res[0]) <= res[1]:
        return ls[0], 0
    else:
        return ls[0], res[0]


def _parallel_compute(pool, args_list, subs, destination, timeout, permute=False, symbolic_int=False):
    if destination is None:
        return_dict = True
        destination = dict()
    else:
        return_dict = False

    if timeout is False or symbolic_int:
        timeout = None

    if timeout is not True:
        future = pool.map(_apply, args_list, timeout=timeout)
        results = future.result()
        num_args_list = list()
        i = 0
        while True:
            try:
                res = next(results)
                if symbolic_int:
                    expr = res[1].simplify()
                    destination[res[0]] = expr
                    if permute:
                        i = res[0][0]
                        idx = res[0][1:]
                        perm_idx = multiset_permutations(idx)
                        for perm in perm_idx:
                            idx = [i] + perm
                            destination[tuple(idx)] = expr
                else:
                    destination[res[0]] = float(res[1].subs(subs))
            except StopIteration:
                break
            except TimeoutError:
                num_args_list.append(args_list[i] + [subs])
            i += 1
    else:
        num_args_list = [args + [subs] for args in args_list]

    future = pool.map(_num_apply, num_args_list)
    results = future.result()
    if permute:
        while True:
            try:
                res = next(results)
                i = res[0][0]
                idx = res[0][1:]
                perm_idx = multiset_permutations(idx)
                for perm in perm_idx:
                    idx = [i] + perm
                    destination[tuple(idx)] = res[1]
            except StopIteration:
                break
    else:
        while True:
            try:
                res = next(results)
                destination[res[0]] = res[1]
            except StopIteration:
                break

    if return_dict:
        return destination
