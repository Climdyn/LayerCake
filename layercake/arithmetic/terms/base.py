
from abc import ABC, abstractmethod
import sparse as sp
from pebble import ProcessPool as Pool
from concurrent.futures import TimeoutError
from multiprocessing import cpu_count
from sympy.utilities.iterables import multiset_permutations

from scipy.integrate import dblquad
from sympy import ImmutableSparseMatrix, ImmutableSparseNDimArray, lambdify, Lambda
from layercake.utils.operators import evaluate_expr
from layercake.utils.commutativity import enable_commutativity, disable_commutativity


class ArithmeticTerm(ABC):
    """Base class for arithmetic terms"""
    def __init__(self, field, inner_product_definition, name=''):

        self.name = name
        self.inner_products = None
        self.field = field
        self.inner_product_definition = inner_product_definition

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

    @abstractmethod
    def _integrations(self, basis, numerical=False):
        pass

    def compute_inner_products(self, basis, numerical=False, timeout=None, num_threads=None, permute=False):

        if num_threads is None:
            num_threads = cpu_count()

        args_list = self._integrations(basis, numerical)
        nmod = len(basis)
        rank = len(args_list[0][0])
        matrix_shape = (nmod,) * rank

        if numerical:
            res = sp.zeros(matrix_shape, dtype=float, format='dok')
        else:
            res = None
        with Pool(max_workers=num_threads) as pool:
            output = self._parallel_compute(pool, args_list, basis.substitutions, res, timeout,
                                            symbolic_int=not numerical, permute=permute)
        if not numerical:
            self.inner_products = ImmutableSparseMatrix(*matrix_shape, output)
        else:
            self.inner_products = res.to_coo()

    @staticmethod
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
