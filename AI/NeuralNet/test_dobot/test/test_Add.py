# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

from unittest import TestCase, main
from common.layers import AddLayer

class TestAddLayer(TestCase):
    """test class of AddLayer"""

    def setUp(self):
        self.add = AddLayer()

    def tearDown(self):
        del self.add
    
    def test_forward(self):
        """test method of forward"""

        # 入力値
        test_patterns = [
            (0, 0, 0),                    # 0入力
            (1, 2, 3),                    # 整数値どうしの入力
            (1.0, 2.0, 3.0),              # 小数点を含む数値どうしの入力
            (1, 2.0, 3.0),                # 整数値と少数を含む数値どうしの入力
            ("hoge", "hoge", "hogehoge"), # 文字どうしの入力
            ("hoge", 2, "hogehoge"),      # 文字と整数値の入力
        ]

        # テスト
        for x, y, expect_result in test_patterns:
            with self.subTest(x=x, y=y):
                self.assertEquals(self.add.forward(x=x, y=y), expect_result)


if __name__ == "__main__":
    main()    
