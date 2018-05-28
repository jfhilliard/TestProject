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


def split_image(image, split_shape=(8, 8)):
    """Return a list of 8x8 image blocks to be used for further jpeg
    compression steps"""

    # TODO: Replace with skimage.view_as_blocks?
    image_blocks_axis0 = np.split(image, range(8, image.shape[0], 8))
    image_blocks_lists = [np.split(block, range(8, image.shape[1], 8),
                                   axis=1) for block in image_blocks_axis0]

    return sum(image_blocks_lists, [])


def unsplit_image(block_list, shape):
    """Take a list of image blocks and a shape tuple that defines the block
    arrangement and return the reconstructed image.

    Assumes all blocks are the same shape"""
    block_start = 0
    for n in range(shape[0]):
        image_row = np.hstack(block_list[block_start:block_start+shape[0]])
        if 'image' in locals():
            image = np.vstack((image, image_row))
        else:
            image = image_row
        block_start += shape[0]

    return image


def gamma_correct(rgb_image, gamma=0.45):
    """Apply gamma correction and scale values to range 0 to 1

    Input: rgb_image with 8-bit channels (0-255)

    Output: gamma corrected rgb image with floating point values (0-1)
    """

    rgb_prime = rgb_image / 255.0
    return rgb_prime**gamma
