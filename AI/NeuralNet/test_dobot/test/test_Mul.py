# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

from unittest import TestCase, main
from common.layers import MulLayer


class TestMulLayer(TestCase):
    """test class of MulLayer"""

    def setUp(self):
        print('setup')
        self.mul = MulLayer()

    def tearDown(self):
        print('tearDown')
        del self.mul

    def test_forward(self):
        """test method of forward"""

        # 入力値
        test_patterns = [
            (0, 0, 0),                    # 0入力
            (1, 2, 2),                    # 整数値どうしの入力
            (1.0, 2.0, 2.0),              # 小数点を含む数値どうしの入力
            (1, 2.0, 2.0),                # 整数値と少数を含む数値どうしの入力
           # ("hoge", "hoge", "hogehoge"), # 文字どうしの入力
            ("hoge", 2, "hogehoge"),      # 文字と整数値の入力
        ]
        
        # テスト
        for x, y, expect_result in test_patterns:
            with self.subTest(x=x, y=y):
                self.assertEquals(self.mul.forward(x=x, y=y), expect_result)

    def test_backward(self):
        """test method of backward"""

        # 初期値
        forward_patterns = [
            (0, 0),          # 0入力
            (1, 2),          # 整数値どうしの入力
            (1.0, 2.0),      # 小数点を含む数値どうしの入力
            (1, 2.0),        # 整数値と少数を含む数値どうしの入力
        ]

        # 入力値
        test_patterns = [
            (1, 0, 0),
            (1, 1, 2),
            (1.0, 1.0, 2.0),
        ]

        for x, y in forward_patterns:
            _ = self.mul.forward(x=x, y=y)
            
            for dout, expect_result_x, expect_result_y in test_patterns:
                with self.subTest(dout=dout):
                    self.assertEquals(self.mul.backward(dout=dout), (expect_result_x, expect_result_y))

if __name__ == "__main__":
    main()
