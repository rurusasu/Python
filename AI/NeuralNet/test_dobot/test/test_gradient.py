# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

from unittest import TestCase, main
from common.gradient import gradient

import numpy as np


def function():
    x = np.arange(0.0, 20.0, 0.1) # 0から20まで、0.1刻みの配列x
    return np.sum(x**2)


class TestGradient(TestCase):
    """test class of Gradient"""

    #def setUp(self):
        #self.gradient = gradient()

    #def tearDown(self):
        #del self.gradient

    def test_gradient(self):
        """test method of gradient"""

        # 入力値
        test_patterns = [
            (np.array([3.0, 4.0]), np.array([6.0, 8.0])),
        ]

        # テスト
        for x, expect_result in test_patterns:
            with self.subTest(x=x):
                self.assertEquals(gradient(f=function(), x=x), expect_result)


if __name__ == "__main__":
    main()