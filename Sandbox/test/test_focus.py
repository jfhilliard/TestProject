# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pylab as plt
from skimage.data import astronaut
from Sandbox.playground import focus
from Sandbox.playground import util


def defocus_image(img, qpe):
    disp_img = util.iffts2(img)

    return util.ffts2(focus.apply_qpe(disp_img, qpe))


class TestQuadFocus(unittest.TestCase):
    """Test that quad autofocus runs and improves image"""

    def test_quad_focus(self):
        image = np.linalg.norm(np.double(astronaut()), axis=2)
        plt.Figure()
        plt.imshow(image)
        plt.title('Original')
        plt.show()

        blurred_image = defocus_image(image, 200)

        plt.Figure()
        plt.imshow(np.abs(blurred_image))
        plt.title('Blurred')
        plt.show()

        focused_image = focus.quad_focus(blurred_image)

        plt.Figure()
        plt.imshow(np.abs(focused_image))
        plt.title('Focused')
        plt.show()


if __name__ == '__main__':
    unittest.main(verbosity=2)
