# -*- coding: utf-8 -*-

import unittest
from Sandbox import playground


class TestPlayground(unittest.TestCase):
    """Test my playground functions"""

    def test_add_list2(self):
        """Test addition of list of 2 elements"""

        x = [1, 2]
        y = playground.sum_list(x)

        expected_value = 3

        self.assertEqual(y, expected_value)

    def test_add_list3(self):
        """Test addition of list of 3 elements"""

        x = [1, 2, 3]
        y = playground.sum_list(x)

        expected_value = 6

        self.assertEqual(y, expected_value)

    def test_with_tuple(self):
        """Test that it works with a tuple input"""

        x = (1, 2, 3, 4)
        y = playground.sum_list(x)

        expected_value = 10

        self.assertEqual(y, expected_value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
