# -*- coding: utf-8 -*-
import unittest
import numpy as np
from numpy.testing import assert_allclose

from Sandbox.utils.image_utils import split_image, unsplit_image


class TestImageSplitters(unittest.TestCase):
    def setUp(self):
        self.x = np.vstack((np.hstack((np.ones((8, 8)) * 0, np.ones((8, 8)) * 1)),
                            np.hstack((np.ones((8, 8)) * 2, np.ones((8, 8)) * 3))))

        self.expected_y = [np.ones((8, 8)) * n for n in range(4)]

    def test_split_image(self):
        """Test image splitting into 8x8 (or user defined) blocks"""
        y = split_image(self.x)
        for n in range(len(y)):
            assert_allclose(y[n], self.expected_y[n])

        # Test with non-evenly divisible block sizes
        self.x = self.x[:-2, :-2]
        y = split_image(self.x)
        self.expected_y[1] = self.expected_y[1][:, :-2]
        self.expected_y[2] = self.expected_y[2][:-2, :]
        self.expected_y[3] = self.expected_y[3][:-2, :-2]
        for n in range(len(y)):
            assert_allclose(y[n],  self.expected_y[n])

    def test_unsplit_image(self):
        """Test that image can be unsplit back to the original shape"""
        y = split_image(self.x)
        z = unsplit_image(y, (2, 2))

        assert_allclose(z, self.x)

        # Test with different number of blocks per dimension
        y = split_image(self.x, split_shape=(8, 4))
        z = unsplit_image(y, (2, 4))

        assert_allclose(z, self.x)

        # Test with non-evenly divisible block sizes
        self.x = self.x[:-2, :-2]
        y = split_image(self.x)
        z = unsplit_image(y, (2, 2))

        assert_allclose(z, self.x)


if __name__ == '__main__':
    unittest.main(verbosity=2)
