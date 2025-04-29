
from abc import ABC, abstractmethod
import sparse as sp
from pebble import ProcessPool as Pool
from concurrent.futures import TimeoutError
from multiprocessing import cpu_count
from sympy.utilities.iterables import multiset_permutations
from itertools import product

from scipy.integrate import dblquad
from sympy import ImmutableSparseMatrix, ImmutableSparseNDimArray, lambdify, Lambda, symbols
from layercake.utils.operators import evaluate_expr
from layercake.utils.commutativity import enable_commutativity, disable_commutativity


class ArithmeticTerm(ABC):
    """Base class for arithmetic terms"""
    def __init__(self, field, inner_product_definition, name=''):

        self.name = name
        self.inner_products = None
        self.field = field
        self.inner_product_definition = inner_product_definition
        self._rank = None

    @property
    def rank(self):
        if self.inner_products is not None:
            return self.inner_products.shape.__len__() - 1
        else:
            return self._rank

    @property
    @abstractmethod
    def symbolic_expression(self):
        pass

    @property
    @abstractmethod
    def numerical_expression(self):
        pass

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

    @abstractmethod
    def _integrations(self, basis, numerical=False):
        pass

    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):

        if num_threads is None:
            num_threads = cpu_count()

        args_list = self._integrations(basis, numerical)
        nmod = len(basis)
        rank = len(args_list[0][0]) - 1
        matrix_shape = (nmod,) * (rank + 1)

        if numerical:
            res = sp.zeros(matrix_shape, dtype=float, format='dok')
        else:
            res = None
        with Pool(max_workers=num_threads) as pool:
            output = _parallel_compute(pool, args_list, basis.substitutions, res, timeout,
                                       symbolic_int=not numerical, permute=permute)
        if not numerical:
            if self._rank == 1:
                self.inner_products = ImmutableSparseMatrix(*matrix_shape, output)
            else:
                self.inner_products = ImmutableSparseNDimArray(output, matrix_shape)
        else:
            self.inner_products = res.to_coo()


class OperationOnTerms(ABC):
    """Base class for arithmetic terms"""
    def __init__(self, *terms, **kwargs):

        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'rank' in kwargs:
            rank = kwargs['rank']
            if rank is not None:
                self._rank = rank
            else:
                self._rank = len(terms)
        else:
            self._rank = len(terms)
        self._terms = terms
        self.inner_products = None

    @property
    def number_of_terms(self):
        return self._terms.__len__()

    @abstractmethod
    def operation(self, *terms, evaluate=False):
        pass

    @property
    def symbolic_expression(self):
        return self.operation(*self._symbolic_expressions_list, evaluate=False)

    @property
    def numerical_expression(self):
        return self.operation(*self._numerical_expressions_list, evaluate=False)

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
        return Lambda(tuple(ssdc), foo)

    @property
    def symbolic_function(self):
        ss = self._fields_list
        ssdc = list()
        for s in ss:
            sdc = disable_commutativity(s.symbol)
            if sdc not in ssdc:
                ssdc.append(sdc)
        for i, ts in enumerate(self._terms):
            dcexpr = ts.symbolic_expression
            if i == 0:
                foo = dcexpr
            else:
                foo = self.operation(foo, dcexpr)
        return Lambda(tuple(ssdc), foo)


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
        return Lambda(tuple(ssdc), foo)

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
        return Lambda(tuple(ssdc), foo)

    def _integrations(self, *basis, numerical=False):
        if len(basis) == 1:
            nmod = len(basis[0])
            nmodr = (range(nmod),) * (self._rank + 1)
        else:
            if len(basis) != self._rank + 1:
                raise ValueError('The number of basis provided should match the rank of the term.')
            nmod = tuple(map(lambda x: len(x), basis))
            nmodr = list()
            for n in nmod:
                nmodr.append(range(n))
        inner_product = self._terms[0].inner_product_definition.inner_product
        args_list = [(indices, inner_product, self._inner_product_arguments(basis, indices, numerical=numerical))
                     for indices in product(*nmodr)]

        return args_list

    def _inner_product_arguments(self, basis, indices, numerical=False):
        res = list()
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

    def compute_inner_products(self, *basis, numerical=False, timeout=None, num_threads=None, permute=False):

        if num_threads is None:
            num_threads = cpu_count()

        args_list = self._integrations(*basis, numerical=numerical)
        if len(basis) == 1:
            basis = basis[0]
            nmod = len(basis)
            rank = len(args_list[0][0]) - 1
            matrix_shape = (nmod,) * (rank + 1)
            substitutions = basis.substitutions
        else:
            if len(basis) != self._rank + 1:
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
            if self._rank == 1:
                self.inner_products = ImmutableSparseMatrix(*matrix_shape, output)
            else:
                self.inner_products = ImmutableSparseNDimArray(output, matrix_shape)
        else:
            self.inner_products = res.to_coo()


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
