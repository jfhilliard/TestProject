# -*- coding: utf-8 -*-

import unittest
import numpy as np
import scipy.fftpack

from Sandbox.utils.dct import dct, idct


class TestDCT(unittest.TestCase):
    """Test the discrete cosine transform"""
    def setUp(self):
        N = 8
        self.N = N
        x1 = np.ones(N)
        x2 = np.zeros(N)
        x2[0] = 1
        self.x_list = [x1, x2]

        expected1 = np.zeros(N)
        expected1[0] = N
        expected2 = np.cos(np.arange(N) * np.pi / N / 2)
        self.expected_list = [expected1, expected2]

    def test_dct(self):
        """Test DCT with a variety of inputs"""
        for x, expected in zip(self.x_list, self.expected_list):
            y = dct(x)
            np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_vs_scipy_dct(self):
        """Test that my DCT gives same result as scipy"""
        for x in self.x_list:
            y = dct(x)
            y_scipy = scipy.fftpack.dct(x)

            # Scipy has a scale factor of 2 difference
            np.testing.assert_allclose(y, y_scipy / 2.0, atol=1e-10)

    def test_inverse_dct(self):
        """Test that the dct->idct process gets you back to the input."""
        for x in self.x_list:
            y = dct(x)
            x_inv = idct(y)

            np.testing.assert_allclose(x_inv, x, atol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
