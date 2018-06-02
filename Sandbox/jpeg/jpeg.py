# -*- coding: utf-8 -*-

import numpy as np

from Sandbox.utils.image_utils import split_image
from Sandbox.utils.dct import dct2

"""
JPEG algorithm steps
1) Convert RGB to  Y′CBCR
2) Quantize color channels at reduced bit rate
3) Split into 8x8 blocks
4) Run each all channels of each block through DCT
5) Quantize in frequency domain with variable bit rate
6) Perform lossless compression of result with Huffman encoding
"""


class JpegCompressor(object):
    """JPEG Compression class"""
    def __init__(self, rgb_image):
        # Color concversion constants from ITU-R BT.601 specification
        self._k_r = 0.299
        self._k_g = 0.587
        self._k_b = 0.114

        self.rgb_image = rgb_image

    def compress(self):
        """Returns JPEG compressed image data"""
        self.rgb_to_ycbcr()

        blocks_y = split_image(self.ycbcr_image[:, :, 0])
        blocks_cb = split_image(self.ycbcr_image[:, :, 1])
        blocks_cr = split_image(self.ycbcr_image[:, :, 2])

        dct_blocks = []
        for n, blocks in enumerate([blocks_y, blocks_cb, blocks_cr]):
            dct_blocks.append(list(map(dct2, blocks)))

        return dct_blocks

    def rgb_to_ycbcr(self):
        """Converts an 8-bit RGB image to a gamma corrected Y'CBCR Image

        Input: rgb_image with 8-bit channels (0-255)

        Output: ycbcr_image with 8-bit channles (0-255)"""
        ypbpr_image = self.rgb_to_ypbpr()

        ypbpr_image[:, :, 1:3] += 0.5
        ycbcr_image = ypbpr_image * 255

        self.ycbcr_image = ycbcr_image.astype('uint8')

    def rgb_to_ypbpr(self):
        """Converts an RGB image into a Y'PBPR image"""
        rgb_image = self.rgb_image
        if rgb_image.dtype == 'uint8':
            rgb_image = self.gamma_correct(rgb_image)

        # TODO: Implement gamma correction
        r_prime = rgb_image[:, :, 0]
        g_prime = rgb_image[:, :, 1]
        b_prime = rgb_image[:, :, 2]

        # TODO: Implement this as a matrix operation
        y_prime = self._k_r * r_prime + self._k_g * g_prime + self._k_b * b_prime
        c_b = 0.5 * (b_prime - y_prime) / (1.0 - self._k_b)
        c_r = 0.5 * (r_prime - y_prime) / (1.0 - self._k_r)

        component_list = [y_prime, c_b, c_r]

        ycbcr = np.stack(component_list, 2)

        return ycbcr

    @staticmethod
    def gamma_correct(rgb_image, gamma=0.45):
        """Apply gamma correction and scale values to range 0 to 1

        Input: rgb_image with 8-bit channels (0-255)

        Output: gamma corrected rgb image with floating point values (0-1)
        """

        rgb_prime = rgb_image / 255.0
        return rgb_prime**gamma
