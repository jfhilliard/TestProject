# -*- coding: utf-8 -*-

import numpy as np
import skimage.data
import unittest
from numpy.testing import assert_allclose

from Sandbox.jpeg.jpeg import rgb_to_ycbcr, scale_zero_to_one


class TestImageFormatTransforms(unittest.TestCase):
    """Test conversions between RGB and other image array formats"""
    def setUp(self):
        self.data = skimage.data.astronaut()

    def test_rgb_to_ycc(self):
        ycc_out = rgb_to_ycbcr(self.data)

        # Shape should be the same
        self.assertEqual(ycc_out.shape, self.data.shape)

        # Test data correctness
        red_data = self.data
        red_data[:, :, 1:3] = 0
        red_ycbcr = rgb_to_ycbcr(red_data)

        assert_allclose(red_ycbcr[:, :, 1], 0)

    def test_scale_zero_to_one(self):
        data = np.array([3, 4, 5, 6, 9])
        scaled_data = scale_zero_to_one(data)

        self.assertIn(0, scaled_data)
        self.assertIn(1, scaled_data)
        self.assertLessEqual(np.max(scaled_data), 1.0)
        self.assertGreaterEqual(np.min(scaled_data), 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
