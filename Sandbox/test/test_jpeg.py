# -*- coding: utf-8 -*-

import numpy as np
import skimage.data
import unittest
from numpy.testing import assert_allclose

from Sandbox.jpeg.jpeg import (rgb_to_ypbpr, rgb_to_ycbcr, k_r, k_g, k_b,
                               gamma_correct)


class TestImageFormatTransforms(unittest.TestCase):
    """Test conversions between RGB and other image array formats"""
    def setUp(self):
        self.data = skimage.data.astronaut()

    def test_rgb_to_ypbpr(self):
        """Test RGB to Y'PbPr converter"""
        ypbpr_out = rgb_to_ypbpr(self.data)

        # Shape should be the same
        self.assertEqual(ypbpr_out.shape, self.data.shape)

        # Test Y'PbPr range of values
        self.assertGreaterEqual(np.min(ypbpr_out[:, :, 0]), 0)
        self.assertLessEqual(np.max(ypbpr_out[:, :, 0]), 1)
        self.assertGreaterEqual(np.min(ypbpr_out[:, :, 1:3]), -0.5)
        self.assertLessEqual(np.max(ypbpr_out[:, :, 1:3]), 0.5)

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

    def test_gamma_correction(self):
        """Test gamma correction function"""
        rgb_prime = gamma_correct(self.data)

        self.assertEqual(rgb_prime.shape, self.data.shape)
        self.assertGreaterEqual(np.min(rgb_prime), 0)
        self.assertLessEqual(np.max(rgb_prime), 1)

        # Test different values of gamma
        test_gammas = [.25, .5, .75, 1, 1.25]
        for gamma in test_gammas:
            y = gamma_correct(127, gamma=gamma)

            self.assertAlmostEqual(y, (127 / 255)**gamma)

    def test_rgb_to_ycbcr(self):
        ycbcr_out = rgb_to_ycbcr(self.data)

        # Test size, value ranges, and type
        self.assertEqual(ycbcr_out.shape, self.data.shape)
        self.assertGreaterEqual(np.min(ycbcr_out), 0)
        self.assertLessEqual(np.max(ycbcr_out), 255)
        self.assertEqual(ycbcr_out.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main(verbosity=2)