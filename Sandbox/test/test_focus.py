# -*- coding: utf-8 -*-

import unittest
import numpy as np
from skimage.data import astronaut
from Sandbox import focus
from Sandbox import util


def defocus_image(img, qpe):
    disp_img = util.iffts2(img)

    return util.ffts2(focus.apply_qpe(disp_img, qpe))


class TestQuadFocus(unittest.TestCase):
    """Test that quad autofocus runs and improves image"""

    def test_quad_focus(self):
        """Test that we can blur and image and refocus it"""
        image = np.linalg.norm(np.double(astronaut()), axis=2)

        blurred_image = defocus_image(image, 200)

        focused_image = focus.quad_focus(blurred_image)

        np.testing.assert_allclose(np.abs(focused_image), image, atol=1e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
