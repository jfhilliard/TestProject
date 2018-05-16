# -*- coding: utf-8 -*-

import unittest
import numpy as np
from numpy.testing import assert_allclose

from Sandbox.utils.ffts import ffts, iffts, ffts2, iffts2


class TestFFTUtils(unittest.TestCase):
    def test_ffts_odd(self):
        """Test FFTS with odd length"""
        x = [1, 1, 1]
        y = ffts(x)

        expected_y = [0, 3, 0]

        assert_allclose(y, expected_y)

    def test_ffts_even(self):
        """Test FFTS with even length"""
        x = [1, 1, 1, 1]
        y = ffts(x)

        expected_y = [0, 0, 4, 0]

        assert_allclose(y, expected_y)

    def test_iffts_odd(self):
        """Test IFFTS with odd length"""
        x = [1, 1, 1]
        y = iffts(x)

        expected_y = [0, 1, 0]

        assert_allclose(y, expected_y)

    def test_iffts_even(self):
        """Test IFFTS with even length"""
        x = [1, 1, 1, 1]
        y = iffts(x)

        expected_y = [0, 0, 1, 0]

        assert_allclose(y, expected_y)

    def test_1d_inverses(self):
        """Test that FFTS and IFFTS are inverses of each other"""

        def check_inverses(x):
            y = ffts(x)
            z = iffts(y)

            assert_allclose(x, z)

            y = iffts(x)
            z = ffts(y)

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
        """Test FFTS2 with odd x odd input"""
        x = np.ones((3, 3))
        y = ffts2(x)

        expected_y = [[0, 0, 0],
                      [0, 9, 0],
                      [0, 0, 0]]

        assert_allclose(y, expected_y)

    def test_ffts2_odd_even(self):
        """Test FFTS2 with odd x even input"""
        x = np.ones((3, 4))
        y = ffts2(x)

        expected_y = [[0, 0, 0, 0],
                      [0, 0, 12, 0],
                      [0, 0, 0, 0]]

        assert_allclose(y, expected_y)

        y = ffts2(x.T)

        assert_allclose(y.T, expected_y)

    def test_ffts2_even_even(self):
        """Test FFTS2 with even x even input"""
        x = np.ones((4, 4))
        y = ffts2(x)

        expected_y = [[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 16, 0],
                      [0, 0, 0, 0]]

        assert_allclose(y, expected_y)

    def test_iffts2_odd_odd(self):
        """Test IFFTS2 with odd x odd input"""
        x = np.ones((3, 3))
        y = iffts2(x)

        expected_y = [[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]]

        assert_allclose(y, expected_y)

    def test_iffts2_odd_even(self):
        """Test IFFTS2 with even x even input"""
        x = np.ones((3, 4))
        y = iffts2(x)

        expected_y = [[0, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0]]

        assert_allclose(y, expected_y)

        y = iffts2(x.T)

        assert_allclose(y.T, expected_y)

    def test_iffts2_even_even(self):
        """Test IFFTS2 with even x even input"""
        x = np.ones((4, 4))
        y = iffts2(x)

        expected_y = [[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0]]

        assert_allclose(y, expected_y)

    def test_2d_inverses(self):
        """Test that FFTS and IFFTS are inverses of each other"""
        def check_inverses(x):
            y = ffts2(x)
            z = iffts2(y)
            assert_allclose(x, z)

            y = iffts2(x)
            z = ffts2(y)
            assert_allclose(x, z)

        x = np.ones((3, 3))
        check_inverses(x)

        x = np.ones((3, 4))
        check_inverses(x)

        x = np.ones((4, 4))
        check_inverses(x)


if __name__ == '__main__':
    unittest.main(verbosity=2)
