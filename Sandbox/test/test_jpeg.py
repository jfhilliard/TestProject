# -*- coding: utf-8 -*-

import numpy as np
import skimage.data
import unittest
from numpy.testing import assert_allclose

from Sandbox.jpeg.jpeg import JpegCompressor


class TestImageFormatTransforms(unittest.TestCase):
    """Test conversions between RGB and other image array formats"""
    def setUp(self):
        self.data = skimage.data.astronaut()

    def test_rgb_to_ypbpr(self):
        """Test RGB to Y'PbPr converter"""
        jpeg = JpegCompressor()
        ypbpr_out = jpeg.rgb_to_ypbpr(self.data)

        # Shape should be the same
        self.assertEqual(ypbpr_out.shape, self.data.shape)

        # Test Y'PbPr range of values
        self.assertGreaterEqual(np.min(ypbpr_out[:, :, 0]), 0)
        self.assertLessEqual(np.max(ypbpr_out[:, :, 0]), 1)
        self.assertGreaterEqual(np.min(ypbpr_out[:, :, 1:3]), -0.5)
        self.assertLessEqual(np.max(ypbpr_out[:, :, 1:3]), 0.5)

        k_r = jpeg._k_r
        k_g = jpeg._k_g
        k_b = jpeg._k_b

        # Test data correctness (Red)
        red_rgb = np.array([[[1, 0, 0]]])
        jpeg = JpegCompressor()
        red_ycbcr = jpeg.rgb_to_ypbpr(red_rgb)
        assert_allclose(red_ycbcr, [[[k_r, -0.5 * k_r / (1 - k_b), 0.5]]])

        # Test data correctness (Green)
        green_rgb = np.array([[[0, 1, 0]]])
        jpeg = JpegCompressor()
        green_ycbcr = jpeg.rgb_to_ypbpr(green_rgb)
        assert_allclose(green_ycbcr, [[[k_g, -0.5 * k_g / (1 - k_b),
                                        -0.5 * k_g / (1 - k_r)]]])
        # Test data correctness (Blue)
        blue_rgb = np.array([[[0, 0, 1]]])
        jpeg = JpegCompressor()
        blue_ycbcr = jpeg.rgb_to_ypbpr(blue_rgb)
        assert_allclose(blue_ycbcr, [[[k_b, 0.5, -0.5 * k_b / (1 - k_r)]]])

        # Test data correctness (White)
        white_rgb = np.array([[[1, 1, 1]]])
        jpeg = JpegCompressor()
        white_ycbcr = jpeg.rgb_to_ypbpr(white_rgb)
        assert_allclose(white_ycbcr, [[[1, 0, 0]]], atol=1e-10)

    def test_gamma_correction(self):
        """Test gamma correction function"""
        jpeg = JpegCompressor()
        rgb_prime = jpeg.gamma_correct(self.data)

        self.assertEqual(rgb_prime.shape, self.data.shape)
        self.assertGreaterEqual(np.min(rgb_prime), 0)
        self.assertLessEqual(np.max(rgb_prime), 1)

        # Test different values of gamma
        test_gammas = [.25, .5, .75, 1, 1.25]
        for gamma in test_gammas:
            y = jpeg.gamma_correct(127, gamma=gamma)

            self.assertAlmostEqual(y, (127 / 255)**gamma)

    def test_rgb_to_ycbcr(self):
        jpeg = JpegCompressor()
        ycbcr_image = jpeg.rgb_to_ycbcr(self.data)

        # Test size, value ranges, and type
        self.assertEqual(ycbcr_image.shape, self.data.shape)
        self.assertGreaterEqual(np.min(ycbcr_image), 0)
        self.assertLessEqual(np.max(ycbcr_image), 255)
        self.assertEqual(ycbcr_image.dtype, np.uint8)


class TestJpegCompressor(unittest.TestCase):
    """Test the JPEG compression chain"""
    def setUp(self):
        self.data = skimage.data.astronaut()

    def test_compress(self):
        """Test the compression chain"""
        jpeg = JpegCompressor()
        compressed = jpeg.compress(self.data)

        # Test that all the data types are correct
        self.assertIsInstance(compressed, np.ndarray)
        for elem in compressed:
            self.assertIsInstance(elem, np.ndarray)

        for chan in range(3):
            block_0 = compressed[chan][0]
            self.assertEqual(block_0.shape, (8, 8))
            self.assertEqual(block_0.dtype.kind, 'i')

        # Test that the compressed data is equal or smaller than the input
        self.assertLessEqual(compressed.size, self.data.size)

        # TODO: Need a test to check value correctness


class TestJpegDecompressor(unittest.TestCase):
    """Test that the decompression chain inverts the compression chain"""
    def setUp(self):
        self.data = skimage.data.astronaut()

    def test_decompress(self):
        jpeg = JpegCompressor()
        compressed = jpeg.compress(self.data)
        decompressed = jpeg.decompress(compressed)

        assert_allclose(decompressed, self.data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
