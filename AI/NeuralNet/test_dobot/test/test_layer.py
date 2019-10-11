#cording: utf-8

import sys, os
sys.path.append(os.getcwd())
import numpy as np

# テストしたい関数をセット
from unittest import TestCase, main
from common.layers import*


class TestLayers(TestCase):
    """test class of Layers.py
    """

    def setUp(self):
        print('setup')
        self.mul = mulLayer()
        self.add = addLayer()
        self.relu = relu()
        self.sigmoid = sigmoid()
        
        
    def test_mullayer(self):
        """test method of MulLayer"""
        # 実際に入力する値
        val = 10
        matrix_x = np.array([[-2, -1, 0], [0, 1, 2]])
        matrix_y = np.array([[2, 1, 0], [0, -1, -2]])

        # テストを行う関数をセット
        _ = self.mul.forward(matrix_x, matrix_y)
        actual_matrix = self.mul.backward(val)
        self.assertEqual(len(actual_matrix), 2)

    def test_addlayer(self):
        """test method of AddLayer"""
        # 実際に入力する値
        val = 10
        matrix_x = np.array([[-2, -1, 0], [0, 1, 2]])
        matrix_y = np.array([[2, 1, 0], [0, -1, -2]])

        # テストを行う関数をセット
        _ = self.add.forward(matrix_x, matrix_y)
        actual_matrix = self.add.backward(val)
        self.assertEqual(len(actual_matrix), 2)

    def test_relulayer(self):
        """test method of ReluLayer"""
        # 実際に入力する値
        matrix_x = np.array([[-2, -1, 0], [0, 1, 2]])

        # テストを行う関数をセット
        actual_matrix_forward = self.relu.forward(matrix_x)
        actual_matrix_backward = self.relu.backward(matrix_x)
        
        self.assertEqual(actual_matrix_forward.max(), 2)
        self.assertEqual(actual_matrix_forward.min(), 0)
        self.assertEqual(actual_matrix_backward.max(), 2)
        self.assertEqual(actual_matrix_backward.min(), 0)

    def test_sigmoid(self):
        """test method of SigmoidLayer"""
        # 実際に入力する値
        val = 1000

        # テストを行う関数をセット
        actual_val_forward = self.sigmoid.forward(val)
        actual_val_backward = self.sigmoid.backward(val)

        self.assertAlmostEqual(actual_val_forward, 1)
        self.assertAlmostEqual(actual_val_backward, 0)

    def tearDown(self):
        print('tearDown')
        del self.mul
        del self.add
        del self.relu
        del self.sigmoid


class TestAffine_forward(TestCase):
    """test class of AffineLayer"""

    def setUp(self):
        print('setup')
        self.params = {}
        self.params['W'] = np.random.randn(5, 5)
        self.params['b'] = np.zeros(5)
        self.affine = affine(self.params['W'], self.params['b'])
        

    def terDown(self):
        print('terDown')
        del self.params
        del self.affine

    def test_forward(self):
        """test method of Affine_forward"""
        # 入力と出力予想
        test_patterns = [
            (np.ones((3, 5)), (3, 5)),
            # OK
            #(np.ones((3, 1)), (3, 5)),
            # shapes (3,1) and (5,5) not aligned: 1 (dim 1) != 5 (dim 0)
        ]
        # テストを行う関数をセット
        for x, expect_result in test_patterns:
            with self.subTest(x=x):
                 self.assertEqual(self.affine.forward(x=x).shape, expect_result)
    

class TestAffine_backward(TestCase):
    """test class of AffineLayer"""

    def setUp(self):
        print('setup')
        self.params = {}
        self.params['W'] = np.random.randn(5, 5)
        self.params['b'] = np.zeros(5)
        self.x = np.ones((3, 5))
        self.affine = affine(self.params['W'], self.params['b'])
        self.affine.forward(self.x)

    def terDown(self):
        print('terDown')
        del self.params
        del self.x
        del self.affine

    def test_backward(self):
        """test method of Affine_backward"""
        # 入力と出力予想
        test_patterns = [
            (np.ones((3, 5)), (3, 5))
            # W = (5, 5)
            # dW = (5, 3)
            # b = [-0.01 -0.01 -0.01 -0.01 -0.01]
        ]


        # テストを行う関数をセット
        for dout, expect_result in test_patterns:
            with self.subTest(dout=dout):
                self.assertEqual(self.affine.backward(dout).shape, expect_result)


class TestDense(TestCase):
    """test class of DenseLayer"""

    def setUp(self):
        print('setup')
        self.dense = Dense(10) # ユニット数を10に設定
        self.rear_node = 3
        print(self.dense.initializer)

    def tearDown(self):
        print('tearDown')
        del self.dense
        del self.rear_node


    def test__init_weight(self):
        expect = (10, self.rear_node)

        self.dense.compile(self.rear_node)
        self.assertEqual(self.dense.params['W'].shape, expect)

    """
    def test_compile(self):

        expect_result = 2

        # テスト
        for i in range(4):
            with self.subTest(rear_node = (3 + i)):
                self.dense.compile(rear_node)

                self.asserrtEqual(self.dense.params['W'].shape, (3+i, expect_result))
    """



if __name__ == "__main__":
    main()
