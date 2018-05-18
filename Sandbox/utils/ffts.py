# -*- coding: utf-8 -*-

import numpy as np


def ffts(x):

    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))


def iffts(x):

    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x)))


def ffts2(x):

    x_shift = np.fft.ifftshift(x, (0, 1))
    y_shift = np.fft.fft2(x_shift)
    return np.fft.fftshift(y_shift, (0, 1))


def iffts2(x):

    x_shift = np.fft.ifftshift(x, (0, 1))
    y_shift = np.fft.ifft2(x_shift)
    return np.fft.fftshift(y_shift, (0, 1))
