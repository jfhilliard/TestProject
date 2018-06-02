# -*- coding: utf-8 -*-

import numpy as np


def dct(x, axis=0):
    """Return the discrete cosine transform (DCT) of x (uses type 2 DCT)"""
    N = x.shape[axis]
    X = np.zeros(x.shape)
    n = np.arange(N)
    k = np.arange(N)

    if axis == 1:
        X = np.matmul(x, np.cos(np.pi / N * np.outer(n + 0.5, k)))
    elif axis == 0:
        X = np.matmul(x.T, np.cos(np.pi / N * np.outer(n + 0.5, k))).T
    else:
        raise("Unsupported Axis. Only 0 or 1 is valid")

    return X


def idct(x, axis=0):
    """Return the inverse discrete cosine transform (DCT) of x"""
    N = x.shape[axis]
    n = np.arange(1, N)
    k = np.arange(N)

    x1 = np.take(x, n, axis)
    if axis == 1:
        X = np.matmul(x1, np.cos(np.pi / N * np.outer(n, k + 0.5)))
    elif axis == 0:
        X = np.matmul(x1.T, np.cos(np.pi / N * np.outer(n, k + 0.5))).T
    else:
        raise("Unsupported Axis. Only 0 or 1 is valid")
    X += 0.5 * np.take(x, 0, axis=axis)
    X *= 2 / N

    return X


def dct2(x):
    """Returns the 2d DCT of 2d array x"""
    x = dct(x, axis=0)
    return dct(x, axis=1)


def idct2(x):
    """Returns the 2d DCT of 2d array x"""
    x = idct(x, axis=0)
    return idct(x, axis=1)
