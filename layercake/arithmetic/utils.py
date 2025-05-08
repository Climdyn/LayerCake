
from sympy import Mul


def sproduct(*terms, evaluate=False):
    for i, t in enumerate(terms):
        if i == 0:
            res = t
        else:
            res = Mul(res, t, evaluate=evaluate)
    return res
