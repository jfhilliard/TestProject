# -*- coding: utf-8 -*-

import numpy as np

"""
JPEG algorithm steps
1) Convert RGB to  Y′CBCR
2) Quantize color channels at reduced bit rate
3) Split into 8x8 blocks
4) Run each all channels of each block through DCT
5) Quantize in frequency domain with variable bit rate
6) Perform lossless compression of result with Huffman encoding
"""

# Color concversion constants
k_r = 0.299
k_g = 0.587
k_b = 0.114


def rgb_to_ycbcr(rgb_image):
    """Converts an 8-bit RGB image to a gamma corrected Y'CBCR Image

    Input: rgb_image with 8-bit channels (0-255)

    Output: ycbcr_image with 8-bit channles (0-255)"""
    ypbpr_image = rgb_to_ypbpr(rgb_image)

    ypbpr_image[:, :, 1:3] += 0.5
    ycbcr_image = ypbpr_image * 255

    return ycbcr_image.astype('uint8')


def rgb_to_ypbpr(rgb_image):
    """Converts an RGB image into a Y'PBPR image"""
    if rgb_image.dtype == 'uint8':
        rgb_image = gamma_correct(rgb_image)

    # TODO: Implement gamma correction
    r_prime = rgb_image[:, :, 0]
    g_prime = rgb_image[:, :, 1]
    b_prime = rgb_image[:, :, 2]

    # TODO: Implement this as a matrix operation
    y_prime = k_r * r_prime + k_g * g_prime + k_b * b_prime
    c_b = 0.5 * (b_prime - y_prime) / (1.0 - k_b)
    c_r = 0.5 * (r_prime - y_prime) / (1.0 - k_r)

    component_list = [y_prime, c_b, c_r]

    ycbcr = np.stack(component_list, 2)

    return ycbcr


def gamma_correct(rgb_image, gamma=0.45):
    """Apply gamma correction and scale values to range 0 to 1

    Input: rgb_image with 8-bit channels (0-255)

    Output: gamma corrected rgb image with floating point values (0-1)
    """

    rgb_prime = rgb_image / 255.0
    return rgb_prime**gamma
