
"""

    Parallel computations utility module
    ====================================

    Defines functions to deal with parallel computation tasks.

    Warnings
    --------
    Those are low-level computation routines which are not user-friendly.
    Usage must be thoroughly tested.

"""

import threading
import _thread as thread

from concurrent.futures import TimeoutError
from pebble import ProcessExpired
from sympy.utilities.iterables import multiset_permutations
from layercake.utils.integration import symbolic_integration, numerical_integration


def parallel_integration(pool, args_list, substitutions, destination, timeout, permute=False, symbolic_int=False):
    """Functions to integrate |Sympy| expressions, either symbolically or numerically, in parallel.

    Parameters
    ----------
    pool: pebble.ProcessPool
        A Pebble pool of workers.
    args_list: list(tuple)
        A list of tuples with the following arguments for the integration subfunctions:

        * `indices`: Tuple of integers labelling the integrations in the integration queue.
          Will be returned by the worker.
        * `integrals_definition`: A callable returning the integral(s) as a |Sympy| expression.
        * `integrals_arguments`: A tuple with the arguments to be provided to the `integrals_definition` callable.
    substitutions: list(tuple)
        List of 2-tuples containing extra symbolic substitutions to be made at the end of the integral computation.
        The 2-tuples contain first a |Sympy|  expression and then the value to substitute.
    destination: None or sparse.DOK or ~numpy.ndarray
        Place where to store the output. If an array is provided, it will append the output of the integrations to it.
        If `None`, it will create a new dictionary and return it.
        If `symbolic_int` is `True`, then `destination` should be `None`.
    timeout: None or bool or int
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
    permute: bool, optional
        Permute the indices provided, except the first one, and return the result for all these indices.
        Default to `False`.
    symbolic_int: bool, optional
        Force symbolic integration and do not substitute the substitutions at the end, making the output a list of |Sympy| expressions.
        Default to `False`.

    Returns
    -------
    tuple(2-tuple)
        A list with the results, as 2-tuple with the labelling indices and the output of the integration, either as a float or as
        a |Sympy| integration.

    """
    if destination is None:
        return_dict = True
        destination = dict()
    else:
        return_dict = False

    if timeout is False or symbolic_int:
        timeout = None

    if timeout is not True:
        new_args_list = [tuple(list(args)+[substitutions]) for args in args_list]
        future = pool.map(symbolic_integration, new_args_list, timeout=timeout)
        results = future.result()
        num_args_list = list()
        i = 0
        while True:
            try:
                res = next(results)
                if symbolic_int:
                    expr = res[1].simplify()
                    if permute:
                        i = res[0][0]
                        idx = res[0][1:]
                        perm_idx = multiset_permutations(idx)
                        for perm in perm_idx:
                            idx = [i] + perm
                            destination[tuple(idx)] = expr
                    else:
                        destination[res[0]] = expr
                else:
                    destination[res[0]] = float(res[1].subs(substitutions))
                    # permutations missing here ?
            except StopIteration:
                break
            except TimeoutError:
                num_args_list.append(args_list[i] + [substitutions])
            except ProcessExpired as e:
                print("%s. Exit code: %d" % (e, e.exitcode))
            except Exception as e:
                print("Function raised %s" % e)
                print(e.traceback)  # Python's traceback of remote process

            i += 1
    else:
        num_args_list = [list(args) + [substitutions] for args in args_list]

    future = pool.map(numerical_integration, num_args_list)
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


def _inner_product_arguments(args):
    return args[1], args[3], args[4]._inner_product_arguments(args[0], args[1], args[2])


def parallel_symbolic_evaluation(pool, indices_list, inner_product, basis, numerical, term):
    """Functions to evaluate |Sympy| inner products expressions inside tensors in parallel.

    Parameters
    ----------
    pool: concurrent.futures.ThreadPoolExecutor or concurrent.futures.ProcessPoolExecutor
        A pool of workers.
    indices_list: list(tuple)
        A list of tuples with the indices of the tensor entries.
    inner_product: callable
        A callable defining the inner products.
    basis: list(SymbolicBasis)
        List of symbolic function basis on which each element of the term(s) inner products must be decomposed.
    numerical: bool
        Whether to compute numerical or symbolic inner products.
    term: ArithmeticTerms
        Arithmetic term(s) from which the inner products will be computed.

    Returns
    -------
    tuple(2-tuple)
        A list with the results, as 2-tuple with the labelling indices and the output of the evaluation.

    """
    args_list_in = [(basis, indices, numerical, inner_product, term) for indices in indices_list]

    return pool.map(_inner_product_arguments, args_list_in)


def _quit_function(fn_name):
    thread.interrupt_main()


def exit_after(s):
    """
    use as decorator to exit process if 
    function takes longer than s seconds
    """
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, _quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer