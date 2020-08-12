# coding: utf-8

import csv
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

class csvIO:
    def twoD_array(self, x):
        """Convert 2D array"""
        array = [row for row in x]

        return array

    def open_twoD_array(self, filename):
        """Open csv file and Convert 2D array"""

        with open(filename, 'r', encoding='utf_8', errors='', newline='') as f:
            x = csv.reader(f)
            array = self.twoD_array(x)
    
        return array


    #----------------------------------------------------------
    # 取得した配列をデータ変換する関数群
    #----------------------------------------------------------
    def twoD_IntToStr(self, x):
        array = self.twoD_array(x)
        int_array = [[int(v) for v in row] for row in array]

        return int_array


    def twoD_FroatToStr(self, x, digit):
        array = self.twoD_array(x)
        froat_array = [[self.__roundOff__(v, digit) for v in row] for row in array]

        return froat_array

    
    def twoD_Numpy(self, x):
        """Convert array to numpy"""
        x = self.twoD_FroatToStr(x, digit=0.01)
        return np.array(x)

    #--------------------------------------------------------------
    # 特定の配列を取得する関数群
    #--------------------------------------------------------------
    def GetOneRow(self, x, row_number):
        """Get 1 row of values from 2D array"""
        
        return x[row_number]


    def GetMultiRow(self, x, row_range_bottom, row_range_top = 0):
        """Get Multi row of values from 2D array"""

        row_list = []
        if row_range_top >= row_range_bottom:
            return

        for num in range(row_range_top, row_range_bottom):
            row_list.append(self.GetOneRow(x, num))
        
        return row_list


    def GetOneCol(self, x, col_number):
        """Get 1 col of values from 2D array"""

        array = [row[col_number] for row in x]

        return array


    def GetMultiCol(self, x, col_range_end, col_range_first=0):
        """Get Multi col of values from 2D array"""

        col_list = []
        if col_range_first > col_range_end:
            return
        elif col_range_first == col_range_end:
            return self.GetOneCol(x, col_range_first)
        elif col_range_first < col_range_end:
            for row in x:
                col_list.append([row[j] for j in range(col_range_first, col_range_end)])
            return col_list


    def Get_AnytwoD_array(self, x, row_range_top=0, row_range_bottom=0, col_range_first=0, col_range_end=0):
        """Get Any 2D array"""

        # エラーチェック
        if (row_range_top > row_range_bottom):
            print('行のrange開始点が終了点を超えています。')
            return
        """
        if (col_range_first> col_range_end):
            print('列のrange開始点が終了点を超えています。')
            return
        """

        if (row_range_bottom == 0 and col_range_end == 0):  
            matrix = self.twoD_array(x)  
        elif (row_range_bottom != 0 and col_range_end == 0):
            matrix = self.GetMultiRow(x, row_range_bottom, row_range_top)
        elif (row_range_bottom == 0 and col_range_end != 0):
            matrix = self.GetMultiCol(x, col_range_end, col_range_first)
        else:
            v_row = self.GetMultiRow(x, row_range_bottom=row_range_bottom, row_range_top=row_range_top)
            matrix = self.GetMultiCol(v_row, col_range_end=col_range_end, col_range_first=col_range_first)

        return matrix


    #--------------------------------------------------------------------------
    # 四捨五入を行う関数
    #--------------------------------------------------------------------------
    def __roundOff__(self, x, digit):
        """ある桁数(digit)で四捨五入を行う"""
        y = Decimal(float(x)).quantize(Decimal(str(digit)), rounding=ROUND_HALF_UP)
        
        return float(y)


    #--------------------------------------------------------------------------
    # ファイルへの書き出しを行う関数
    #--------------------------------------------------------------------------

    def csv_write(self, filename, data):
        """Write Data to csv file"""
        if (data == None):  # 書き込むデータが無いとき
            return
        with open(filename, 'w', encoding='utf_8', errors='', newline='') as f:

            if(self.wirte_content(f, data) == None):
                print('書き込みが完了しました。')


    def wirte_content(self, f, data):
        """write content"""

        error = 1  # エラーチェック用変数
        witer = csv.writer(f, lineterminator='\n')
        error = witer.writerows(data)

        return error  # エラーが無ければNoneを返す



if __name__ == '__main__':
    io = csvIO()
    
    # ファイルを開く
    #file = io.csv_open('./data.csv', 'r')
    #file_data = csv.reader(file)
    
    # 2D arrayにデータを変換
    #v = io.twoD_array(file_data)
    v = io.open_twoD_array('./data/data.csv')
    # float型の変数に2D arrayのdata typeを変換
    v = io.twoD_FroatToStr(v, 0.01)

    # 2D arrayから行を取得する
    #data_row = io.GetOneRow(v, 1)
    #print(data_row)
    #data_multi_row = io.GetMultiRow(v, 5)
    #print(data_multi_row)

    # 2D arrayから列を取得する
    #data_col = io.GetOneCol(v, 2)
    #print(data_col)
    #data_multi_col = io.GetMultiCol(v, 4)
    #print(data_multi_col)
    
    #v = io.twoD_Numpy(v)
    #array = io.twoD_Numpy(file)
    #print(type(array))

    # 任意の2D arrayを取得する
    array = io.Get_AnytwoD_array(v, col_range_end=3)
    #print(array)

    # ファイルに書き込む
    io.csv_write('./data/test.csv', array)
