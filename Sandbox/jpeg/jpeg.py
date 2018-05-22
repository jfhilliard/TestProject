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


def rgb_to_ypbpr(rgb_image):
    """Converts an RGB image into a Y'CBCR image"""
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
