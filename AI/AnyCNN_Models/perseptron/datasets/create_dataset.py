import sys

sys.path.append(".")
sys.path.append("..")

import pandas as pd
from sklearn.datasets import *


def SKLearnDataSetsLoad(DataSetName):
    if DataSetName is "boston":
        data = load_boston()

    df = pd.DataFrame(data.data, columns=data.feature_names)

    return df


def CreateDataSetCSV(df, path):
    try:
        df.to_csv(path)
    except:
        print("CSVファイルに書き込むことができませんでした。")


if __name__ == "__main__":
    import os

    path = os.path.dirname(os.path.abspath(__file__))
    # print(path)
    name = "boston"
    path = path + os.sep + name + ".csv"
    df = SKLearnDataSetsLoad(name)
    # print(df.tail())
    CreateDataSetCSV(df, path)
