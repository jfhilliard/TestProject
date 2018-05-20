# -*- coding: utf-8 -*-

import unittest
import numpy as np
import scipy.fftpack

from numpy.testing import assert_allclose
from Sandbox.utils.dct import dct, idct, dct2


class TestDCT(unittest.TestCase):
    """Test the discrete cosine transform"""
    def setUp(self):
        N = 8
        self.N = N
        x1 = np.ones(N)
        x2 = np.zeros(N)
        x2[0] = 1
        x_cos = [np.cos((np.arange(N) + 0.5) * x * np.pi / N) for x in range(N)]
        self.x_list = [x1, x2] + x_cos

        expected1 = np.zeros(N)
        expected1[0] = N
        expected2 = np.cos(np.arange(N) * np.pi / N / 2)
        expected_cos = np.eye(N) * N / 2.0
        expected_cos[0, 0] = N
        self.expected_list = [expected1, expected2] + list(expected_cos)

    def test_dct(self):
        """Test DCT with a variety of inputs"""
        for x, expected in zip(self.x_list, self.expected_list):
            y = dct(x)
            assert_allclose(y, expected, atol=1e-10)

    def test_vs_scipy_dct(self):
        """Test that my DCT gives same result as scipy"""
        for x in self.x_list:
            y = dct(x)
            y_scipy = scipy.fftpack.dct(x)

            # Scipy has a scale factor of 2 difference
            assert_allclose(y, y_scipy / 2.0, atol=1e-10)

    def test_inverse_dct(self):
        """Test that the dct->idct process gets you back to the input."""
        for x in self.x_list:
            y = dct(x)
            x_inv = idct(y)

            assert_allclose(x_inv, x, atol=1e-10)


class TEST2DDCT(unittest.TestCase):
    """Test the 2D discrete cosine transform"""
    def setUp(self):
        N = 8
        self.N = N

        def create_expected(n, m):
            # Creates expected output for 2d cosine with frequencies at m and n
            x = np.zeros((N, N))
            if n == 0:
                if m == 0:
                    x[n, m] = N**2
                else:
                    x[n, m] = N * (N / 2)
            elif m == 0:
                x[n, m] = N * (N / 2)
            else:
                x[n, m] = (N / 2)**2
            return x

        # Create some test data with the same freqency in each dimension
        x_cos1 = [np.cos((np.arange(N) + 0.5) * x * np.pi / N)
                  for x in range(N)]

        self.x_cos2_list = [np.outer(x_cos, x_cos) for x_cos in x_cos1]
        self.expected_cos2_list = [create_expected(n, n) for n in range(N)]

        # Create some test data with different frequencies in each dimension
        y_cos1 = [np.cos((np.arange(N) + 0.5) * x * np.pi / N)
                  for x in range(N-1, -1, -1)]

        self.x_cos2_list += [np.outer(x_cos, y_cos)
                             for x_cos, y_cos in zip(x_cos1, y_cos1)]

        self.expected_cos2_list += [create_expected(n, m)
                                    for n, m in enumerate(range(N-1, -1, -1))]

    def test_dct2(self):
        for n, x in enumerate(self.x_cos2_list):
            y = dct2(x)
            assert_allclose(y, self.expected_cos2_list[n], atol=1e-10)

    def test_vs_scipy_dct2(self):
        """Test that my DCT2 gives same result as scipy"""
        for x in self.x_cos2_list:
            y = dct2(x)
            y_scipy = scipy.fftpack.dctn(x)

            # Scipy has a scale factor of 2^2 difference
            assert_allclose(y, y_scipy / (2.0**2.0), atol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)