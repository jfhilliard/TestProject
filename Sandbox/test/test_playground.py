# -*- coding: utf-8 -*-

import unittest
from Sandbox.playground import playground


class TestPlayground(unittest.TestCase):
    """Test my playground functions"""

    def test_add_list2(self):
        """Test addition of list of 2 elements"""

        x = [1, 2]
        y = playground.sum_list(x)

        expected_value = 3

        self.assertEqual(y, expected_value)


if __name__ == '__main__':
    unittest.main()
