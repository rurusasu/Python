import sys
sys.path.append('.')
sys.path.append('..')

import pandas as pd


def readDataFromExcel(file_path: str, target_page: int = 1):
    '''
    指定されたfile_pathのxlsxファイルからデータを読み込み、
    DataFrameに変換する。

    Params
    ------
    file_path (str):
        読み込むファイルパス
    target_page (int):
        読み取るページ番号 1以上の整数

    Return
    ------
    df (Pandas DataFrame):
    '''

    if type(target_page) is int:
        target_page -= 1

        df = pd.read_excel(file_path,
                           sheet_name=target_page,
                           header=0,
                           index_col=0)
    else:
        raise ValueError('invaild number')

    return df


if __name__ == "__main__":
    import os
    from config.config import cfg

    f_path = cfg.TEST_DIR+os.sep+'result.xlsx'
    readDataFromExcel(f_path, 2)
