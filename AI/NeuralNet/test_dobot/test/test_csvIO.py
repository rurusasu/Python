# cording: utf-8

import os
from csvIO import csvIO
from io import StringIO
from unittest import TestCase, main

import csv

class TestCSVIO(TestCase):
    """Test class of csvIO"""

    def setUp(self):
        self.io = csvIO()
    
    def tearDown(self):
        del self.io

    def test_csv_write(self):
        """test method of csv_write"""

        # 入力値
        
        test_patterns = [[100,100.0],"hoge"]   
        #AssertionError: [[100, 100.0], 'hoge'] != '100,100.0\nh,o,g,e\n'
        # 文字列は1文字ずつに分割されて保存される

        """
        test_patterns = [100,100.0, 'hoge']
        _csv.Error: iterable expected, not int
        """

        """
        test_patterns = [[100], 100.0, 'hoge'] 
        _csv.Error: iterable expected, not float
        """

        """
        test_patterns = [[100], [100.0], 'hoge']
        AssertionError: [[100], [100.0], 'hoge'] != '100\n100.0\nh,o,g,e\n'
        """
        
        filename = 'foo_test.csv'

        # テスト
        try:
            self.io.csv_write(filename, test_patterns)

            with open(filename, 'rU') as f:
                text = f.read()

        finally:
            os.remove(filename)

        self.assertEquals(test_patterns, text)

if __name__ == "__main__":
    main()