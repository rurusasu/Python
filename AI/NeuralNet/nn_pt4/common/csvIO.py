# coding: utf-8

import csv
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN


def twoD_array(x):
    """Convert 2D array"""
    array = [row for row in x]

    return array

def open_twoD_array(filename):
    """Open csv file and Convert 2D array"""

    with open(filename, 'r', encoding='utf_8', errors='', newline='') as f:
        x = csv.reader(f)
        array = twoD_array(x)
    
    return array


#----------------------------------------------------------
# 取得した配列をデータ変換する関数群
#----------------------------------------------------------
def twoD_IntToStr(x):
    array = twoD_array(x)
    int_array = [[int(v) for v in row] for row in array]

    return int_array


def twoD_FroatToStr(x, digit):
    array = twoD_array(x)
    froat_array = [[__roundOff__(v, digit) for v in row] for row in array]

    return froat_array

    
def twoD_Numpy(x):
        """Convert array to numpy"""
        x = twoD_FroatToStr(x, digit=0.01)
        return np.array(x)

#--------------------------------------------------------------
# 特定の配列を取得する関数群
#--------------------------------------------------------------
def GetOneRow(x, row_number):
    """Get 1 row of values from 2D array"""
    return x[row_number]


def GetMultiRow(x, row_range_bottom, row_range_top = 0):
    """Get Multi row of values from 2D array"""

    row_list = []
    if row_range_top >= row_range_bottom:
        return

    for num in range(row_range_top, row_range_bottom):
        row_list.append(GetOneRow(x, num))
    return row_list


def GetOneCol(x, col_number):
    """Get 1 col of values from 2D array"""

    array = [row[col_number] for row in x]
    return array


def GetMultiCol(x, col_range_end, col_range_first=0):
    """Get Multi col of values from 2D array"""

    col_list = []
    if col_range_first > col_range_end:
        return
    elif col_range_first == col_range_end:
        return GetOneCol(x, col_range_first)
    elif col_range_first < col_range_end:
        for row in x:
            col_list.append([row[j] for j in range(col_range_first, col_range_end)])
        return col_list


def Get_AnytwoD_array(x, row_range_top=0, row_range_bottom=0, col_range_first=0, col_range_end=0):
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
        matrix = twoD_array(x)  
    elif (row_range_bottom != 0 and col_range_end == 0):
        matrix = GetMultiRow(x, row_range_bottom, row_range_top)
    elif (row_range_bottom == 0 and col_range_end != 0):
        matrix = GetMultiCol(x, col_range_end, col_range_first)
    else:
        v_row = GetMultiRow(x, row_range_bottom=row_range_bottom, row_range_top=row_range_top)
        matrix = GetMultiCol(v_row, col_range_end=col_range_end, col_range_first=col_range_first)

    return matrix


    #--------------------------------------------------------------------------
    # 四捨五入を行う関数
    #--------------------------------------------------------------------------
def __roundOff__(x, digit):
    """ある桁数(digit)で四捨五入を行う"""
    y = Decimal(float(x)).quantize(Decimal(str(digit)), rounding=ROUND_HALF_UP)
        
    return float(y)


#--------------------------------------------------------------------------
# ファイルへの書き出しを行う関数
#--------------------------------------------------------------------------
def csv_write(filename, data):
    """Write Data to csv file"""
    if (data == None):  # 書き込むデータが無いとき
        return
    with open(filename, 'w', encoding='utf_8', errors='', newline='') as f:

        if(wirte_content(f, data) == None):
            print('書き込みが完了しました。')


def wirte_content(f, data):
    """write content"""

    error = 1  # エラーチェック用変数
    witer = csv.writer(f, lineterminator='\n')
    error = witer.writerows(data)

    return error  # エラーが無ければNoneを返す


def DataConv(Org_FileName, Out_FileName, col_range_end, col_range_first=0, digit=None):
    """csvIO内の関数を用いて、2次元配列から欲しい行を取り出す。"""
    v = open_twoD_array(Org_FileName)
    if digit != None and digit != str:
        digit = str(10**-int(digit))
        v = twoD_FroatToStr(v, digit=digit)
    
    array = GetMultiCol(v, col_range_first=col_range_first, col_range_end=col_range_end)
    csv_write(Out_FileName, array)
    
    



if __name__ == '__main__':
    # ファイルを開く
    #file = io.csv_open('./data.csv', 'r')
    #file_data = csv.reader(file)
    
    # 2D arrayにデータを変換
    #v = io.twoD_array(file_data)
    v = open_twoD_array('./data/data.csv')
    # float型の変数に2D arrayのdata typeを変換
    v = twoD_FroatToStr(v, 0.01)

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
    array = Get_AnytwoD_array(v, col_range_end=3)
    #print(array)

    # ファイルに書き込む
    csv_write('./data/test.csv', array)
