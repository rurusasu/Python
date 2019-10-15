#cording: utf-8

import sys, os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt

# テストしたい関数をセット
from unittest import TestCase, main
from common.layers import*
from common.squential import*

class TestSquential(TestCase):
    """test class of squential.py"""

    def setUp(self):
        print('setup')
        self.model = Sequential()

    def terDown(self):
        print('terDown')
        del self.model

    def test_add(self):
        test_patterns = [
            (10, {1:10}),
            (5, {1:10, 2:5}),
            (9, {1:10, 2:5, 3:9})
        ]

        for unit, expect in test_patterns:
            self.model.add(Dense(unit))
            self.assertEqual(self.model.units, expect)
            #self.assertEqual(self.model.sequential)

if __name__ == "__main__":
    main()
