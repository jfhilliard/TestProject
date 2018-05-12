# -*- coding: utf-8 -*-
import numpy as np
from Sandbox import util


def apply_qpe(x, qpe):
    k = np.arange(x.shape[0])
    n = 2.0 * (k - np.ceil(x.shape[0] / 2.0)) / x.shape[0]

    phase = qpe * n**2

    return x * np.exp(1j * phase)


def quad_focus(image):

    qpe = 200
    focused = util.ffts2(apply_qpe(util.iffts2(image), -qpe))

    return focused
