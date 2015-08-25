# -*- coding: utf-8 -*-

import math

import benchpy as bp


def factorial_slow(n):
    assert n >= 0
    return 1 if n == 0 else n * factorial_slow(n-1)


def pow_slow(x, n):
    assert n >= 0
    return 1 if n == 0 else x * pow_slow(x, n-1)


if __name__ == "__main__":
    n = 100
    groups = [
        bp.group("factorial(100)", [
            bp.bench("math_!", math.factorial, n),
            bp.bench("slow_!", factorial_slow, n)
        ]),
        bp.group("pow(100, 100)", [
            bp.bench("math^", math.pow, n, n),
            bp.bench("simple^", pow_slow, n, n)
        ])
    ]

    print(bp.run(groups))
