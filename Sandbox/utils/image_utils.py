# -*- coding: utf-8 -*-
import numpy as np


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
