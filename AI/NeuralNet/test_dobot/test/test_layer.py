#cording: utf-8

from pathlib import Path
import sys
sys.path.append(Path.cwd().parent)
import numpy as np
import matplotlib.pyplot as plt

# テストしたい関数をセット
from unittest import TestCase, main
from common.layers import*


class TestLayers(TestCase):
    """test class of Layers.py
    """

    def setUp(self):
        print('setup')
        self.mul = MulLayer()
        self.add = AddLayer()
        self.relu = Relu()
        self.sigmoid = Sigmoid()
        
    def test_mullayer(self):
        """test method of MulLayer"""
        # 実際に入力する値
        val = 10
        matrix_x = np.array([[-2, -1, 0], [0, 1, 2]])
        matrix_y = np.array([[2, 1, 0], [0, -1, -2]])

        # テストを行う関数をセット
        _ = self.mul.forward(matrix_x, matrix_y)
        actual_matrix = self.mul.backward(val)
        self.assertEquals(len(actual_matrix), 2)

    def test_addlayer(self):
        """test method of AddLayer"""
        # 実際に入力する値
        val = 10
        matrix_x = np.array([[-2, -1, 0], [0, 1, 2]])
        matrix_y = np.array([[2, 1, 0], [0, -1, -2]])

        # テストを行う関数をセット
        _ = self.add.forward(matrix_x, matrix_y)
        actual_matrix = self.add.backward(val)
        self.assertEquals(len(actual_matrix), 2)

    def test_relulayer(self):
        """test method of ReluLayer"""
        # 実際に入力する値
        matrix_x = np.array([[-2, -1, 0], [0, 1, 2]])

        # テストを行う関数をセット
        actual_matrix_forward = self.relu.forward(matrix_x)
        actual_matrix_backward = self.relu.backward(matrix_x)
        
        self.assertEqual(actual_matrix_forward.max(), 2)
        self.assertEquals(actual_matrix_forward.min(), 0)
        self.assertEquals(actual_matrix_backward.max(), 2)
        self.assertEquals(actual_matrix_backward.min(), 0)

    def test_sigmoid(self):
        """test method of SigmoidLayer"""
        # 実際に入力する値
        val = 1000

        # テストを行う関数をセット
        actual_val_forward = self.sigmoid.forward(val)
        actual_val_backward = self.sigmoid.backward(val)

        self.assertAlmostEquals(actual_val_forward, 1)
        self.assertAlmostEquals(actual_val_backward, 0)

    def tearDown(self):
        print('tearDown')
        del self.mul
        del self.add
        del self.relu
        del self.sigmoid


class TestAffineLayer(TestCase):
    """test method of AffineLayer"""

    def setUp(self):
        print('setup')
        self.affine = Affine()
        

    def terDown(self):
        print('terDown')
        del self.affine

    def test_forward(self):
        # 実際に入力する値
        test_patterns = [
            (1, 2, 3),
            ("hoge", 2, "hogehoge"),
            ("hoge", "hoge", "hogehoge"),
        ]

        # テストを行う関数をセット
        for x, y, expect_result in test_patterns:
            with self.subTest(x=x, y=y):
                self.assertEquals(self.affine(x=x, y=y), expect_result)

if __name__ == "__main__":
    main()
