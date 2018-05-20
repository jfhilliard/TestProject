# -*- coding: utf-8 -*-

from math import cos, pi


def dct(x):
    """Return the discrete cosine transform (DCT) of x (uses type 2 DCT)"""
    N = len(x)
    X = [0] * N
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * cos(pi / N * (n + 0.5) * k)

    return X


def idct(x):
    N = len(x)
    X = [0.5 * x[0]] * N
    for k in range(N):
        for n in range(1, N):
            X[k] += x[n] * cos(pi * n / N * (k + 0.5))
        X[k] *= 2 / N

    return X
