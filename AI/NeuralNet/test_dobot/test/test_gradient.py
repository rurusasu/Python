# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

from unittest import TestCase, main
from common.gradient import _numerical_gradient_1d

import numpy as np


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


class TestGradient(TestCase):
    """test class of Gradient"""

    #def setUp(self):
        #self.gradient = gradient()

    #def tearDown(self):
        #del self.gradient

    def test__numerical_gradient_1d(self):
        """test method of _numerical_gradient_1d"""

        # 入力値
        test_patterns = [
            (function_2, np.array([3.0, 4.0]), np.array([6.0, 8.0])),
        ]

        # テスト
        for function, x, expect_result in test_patterns:
            with self.subTest(f=function, x=x):
                self.assertEquals(type(_numerical_gradient_1d(f=function, x=x)), type(expect_result))


if __name__ == "__main__":
    main()
