# -*- coding: utf-8 -*-

import unittest
import numpy as np
from numpy.testing import assert_allclose

from Sandbox.playground import util


class TestFFTUtils(unittest.TestCase):
    def test_ffts_odd(self):
        x = [1, 1, 1]
        y = util.ffts(x)

        expected_y = [0, 3, 0]

        assert_allclose(y, expected_y)

    def test_ffts_even(self):
        x = [1, 1, 1, 1]
        y = util.ffts(x)

        expected_y = [0, 0, 4, 0]

        assert_allclose(y, expected_y)

    def test_iffts_odd(self):
        x = [1, 1, 1]
        y = util.iffts(x)

        expected_y = [0, 1, 0]

        assert_allclose(y, expected_y)

    def test_iffts_even(self):
        x = [1, 1, 1, 1]
        y = util.iffts(x)

        expected_y = [0, 0, 1, 0]

        assert_allclose(y, expected_y)

    def test_1d_inverses(self):

        def check_inverses(x):
            y = util.ffts(x)
            z = util.iffts(y)

            assert_allclose(x, z)

            y = util.iffts(x)
            z = util.ffts(y)

            assert_allclose(x, z)

        x = [1, 1, 1, 1]
        check_inverses(x)

        x = [1, 1, 1]
        check_inverses(x)

        x = [1, 2, 3]
        check_inverses(x)

        x = [1, 1, 2, 3]
        check_inverses(x)

    def test_ffts2_odd_odd(self):
        x = np.ones((3, 3))
        y = util.ffts2(x)

        expected_y = [[0, 0, 0],
                      [0, 9, 0],
                      [0, 0, 0]]

        assert_allclose(y, expected_y)

    def test_ffts2_odd_even(self):
        x = np.ones((3, 4))
        y = util.ffts2(x)

        expected_y = [[0, 0, 0, 0],
                      [0, 0, 12, 0],
                      [0, 0, 0, 0]]

        assert_allclose(y, expected_y)

        y = util.ffts2(x.T)

        assert_allclose(y.T, expected_y)

    def test_ffts2_even_even(self):
        x = np.ones((4, 4))
        y = util.ffts2(x)

        expected_y = [[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 16, 0],
                      [0, 0, 0, 0]]

        assert_allclose(y, expected_y)

    def test_iffts2_odd_odd(self):
        x = np.ones((3, 3))
        y = util.iffts2(x)

        expected_y = [[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]]

        assert_allclose(y, expected_y)

    def test_iffts2_odd_even(self):
        x = np.ones((3, 4))
        y = util.iffts2(x)

        expected_y = [[0, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0]]

        assert_allclose(y, expected_y)

        y = util.iffts2(x.T)

        assert_allclose(y.T, expected_y)

    def test_iffts2_even_even(self):
        x = np.ones((4, 4))
        y = util.iffts2(x)

        expected_y = [[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0]]

        assert_allclose(y, expected_y)

    def test_2d_inverses(self):
        def check_inverses(x):
            y = util.ffts2(x)
            z = util.iffts2(y)
            assert_allclose(x, z)

            y = util.iffts2(x)
            z = util.ffts2(y)
            assert_allclose(x, z)

        x = np.ones((3, 3))
        check_inverses(x)

        x = np.ones((3, 4))
        check_inverses(x)

        x = np.ones((4, 4))
        check_inverses(x)


if __name__ == '__main__':
    unittest.main(verbosity=2)