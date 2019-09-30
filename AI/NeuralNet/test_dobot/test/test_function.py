#cording: utf-8

from pathlib import Path
import sys
sys.path.append(Path.cwd().parent)
import numpy as np
import matplotlib.pyplot as plt

# テストしたい関数をセット
import unittest
from common.functions import*

class Testfunction(unittest.TestCase):
    """ test class of function.py
    """

    def test_step_funciton(self):
        """ test method of step_function
        """
        # 実際に入力する値
        val = 10
        vector = np.array([0, 0, 0])
        matrix = np.array([[0, 0, 0], [0, 0, 0]])
        # 返り値として想定される値
        expected = 1
        # 実際にテストを行う関数をactualにセット
        actual_val    = step_function(val)
        actual_vector = step_function(vector)
        actual_matrix = step_function(matrix)

        self.assertEqual(expected, actual_val)
        self.assertEqual(vector.ndim, actual_vector.ndim)
        self.assertEqual(matrix.ndim, actual_matrix.ndim)


    def test_sigmoid_function(self):
        """ test method of sigmoid_function
        """
        # 実際に入力する値
        vector = np.array([0, 0, 0])
        matrix = np.array([[0, 0, 0], [0, 0, 0]])
        
        # 実際にテストを行う関数をactualにセット
        actual_vector = step_function(vector)
        actual_matrix = step_function(matrix)

        self.assertEqual(vector.ndim, actual_vector.ndim)
        self.assertEqual(matrix.ndim, actual_matrix.ndim)

    

# 結果をグラフ化する
def plot_result(y):
    plt.plot(y)
    plt.title('result')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__=='__main__':
    unittest.main()