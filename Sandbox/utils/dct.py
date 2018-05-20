# -*- coding: utf-8 -*-

import numpy as np
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
    """Return the inverse discrete cosine transform (DCT) of x"""
    N = len(x)
    X = [0.5 * x[0]] * N
    for k in range(N):
        for n in range(1, N):
            X[k] += x[n] * cos(pi * n / N * (k + 0.5))
        X[k] *= 2 / N

    return X


def dct2(x):
    """Returns the 2d DCT of 2d array x"""
    x = np.array(x)
    y = np.zeros(x.shape)
    for n, x_row in enumerate(x):
        y[n] = dct(x_row)
    for n, y_col in enumerate(y.T):
        y[:, n] = dct(y_col)

    return y


def idct2(x):
    """Returns the 2d DCT of 2d array x"""
    x = np.array(x)
    y = np.zeros(x.shape)
    for n, x_row in enumerate(x):
        y[n] = idct(x_row)
    for n, y_col in enumerate(y.T):
        y[:, n] = idct(y_col)

    return y
