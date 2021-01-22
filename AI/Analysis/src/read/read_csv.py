import sys
sys.path.append('.')
sys.path.append('..')

import pandas as pd


def readDataFromCSV(file_path: str):
    '''
    指定されたfile_pathのxlsxファイルからデータを読み込み、
    DataFrameに変換する。

    Params
    ------
    file_path (str):
        読み込むファイルパス

    Return
    ------
    df (Pandas DataFrame):
    '''

    df = pd.read_csv(file_path, header=None)

    return df


if __name__ == "__main__":
    import os
    from config.config import cfg

    f_path = cfg.TEST_DIR+os.sep+'Confusion_Matrix.csv'
    readDataFromCSV(f_path)
