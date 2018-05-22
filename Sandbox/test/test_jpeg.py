# -*- coding: utf-8 -*-

import numpy as np
import skimage.data
import unittest
from numpy.testing import assert_allclose

from Sandbox.jpeg.jpeg import rgb_to_ypbpr, k_r, k_g, k_b


class TestImageFormatTransforms(unittest.TestCase):
    """Test conversions between RGB and other image array formats"""
    def setUp(self):
        self.data = skimage.data.astronaut()

    def test_rgb_to_ypbpr(self):
        ycc_out = rgb_to_ypbpr(self.data)

        # Shape should be the same
        self.assertEqual(ycc_out.shape, self.data.shape)

        # Test data correctness (Red)
        red_rgb = np.array([[[1, 0, 0]]])
        red_ycbcr = rgb_to_ypbpr(red_rgb)
        assert_allclose(red_ycbcr, [[[k_r, -0.5 * k_r / (1 - k_b), 0.5]]])

        # Test data correctness (Green)
        green_rgb = np.array([[[0, 1, 0]]])
        green_ycbcr = rgb_to_ypbpr(green_rgb)
        assert_allclose(green_ycbcr, [[[k_g, -0.5 * k_g / (1 - k_b),
                                        -0.5 * k_g / (1 - k_r)]]])
        # Test data correctness (Blue)
        blue_rgb = np.array([[[0, 0, 1]]])
        blue_ycbcr = rgb_to_ypbpr(blue_rgb)
        assert_allclose(blue_ycbcr, [[[k_b, 0.5, -0.5 * k_b / (1 - k_r)]]])

        # Test data correctness (White)
        white_rgb = np.array([[[1, 1, 1]]])
        white_ycbcr = rgb_to_ypbpr(white_rgb)
        assert_allclose(white_ycbcr, [[[1, 0, 0]]], atol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
