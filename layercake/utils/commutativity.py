
from sympy import Symbol


def enable_commutativity(expr):
    replacements = {s: Symbol(s.name) for s in expr.free_symbols}
    return expr.xreplace(replacements)


def disable_commutativity(expr):
    replacements = {s: Symbol(s.name, commutative=False) for s in expr.free_symbols}
    return expr.xreplace(replacements)


def _remove_key(d, key):
    d.pop(key)
    return d
