from functools import reduce


def identity(x):
    return x


def comp_one(f, g):
    return lambda x: f(g(x))


def comp(*funcs):
    return reduce(comp_one, funcs, identity)


def partial(f, *args):
    return lambda *rest_args: f(*(args + rest_args))
