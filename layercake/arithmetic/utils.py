
from sympy import Mul, Add


def sproduct(*terms, evaluate=False):
    for i, t in enumerate(terms):
        if i == 0:
            res = t
        else:
            res = Mul(res, t, evaluate=evaluate)
    return res


def sadd(*terms, evaluate=False):
    for i, t in enumerate(terms):
        if i == 0:
            res = t
        else:
            res = Add(res, t, evaluate=evaluate)
    return res
