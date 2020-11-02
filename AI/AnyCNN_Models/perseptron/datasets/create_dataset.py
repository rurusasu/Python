import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
import pandas as pd

from sklearn.datasets import *


def SKLearnDataSetsLoad(DataSetName: str):
    """scikit-learnに用意されているデータセットをロードする関数

    Param
    -----
        DataSetName (str):
            ロードするデータセット名
    
    Return
    ------
        data (tuple):
    """
    # アヤメの品種データ
    if DataSetName is "iris":
        data = load_iris()
    # ボストン市郊外の地域別住宅価格データセット
    elif DataSetName is "boston":
        data = load_boston()
    # 糖尿病患者の診断データ
    # 糖尿病患者442人の検査データと、その1年後の疾患進行状況
    elif DataSetName is "diabetes":
        data = load_diabetes()
    # 数字の手書き文字データ
    elif DataSetName is "digits":
        data = load_digits()
    # 生理学的特徴と運動能力の関係についてのデータ
    # 20任の成人男性に対してフィットネスクラブで測定した
    # 3 つの生理学的特徴と 3 つの運動能力の関係を整理したデータセット
    elif DataSetName is "linnerrud":
        data = load_linnerud()
    # ワインの品種データ
    # 11 種類のワインの成分データと、
    # ワイン専門家によるワインの品質がまとめられたデータセット
    elif DataSetName is "wine":
        data = load_wine()
    # 乳がんデータ
    elif DataSetName is "breast_cancer":
        data = load_breast_cancer()
    else:
        print("{} はscikit-learnにありません。".format(DataSetName))
        data = {}

    return data


def CreateDataSetCSV(df, path):
    try:
        df.to_csv(path, header=True, index=False)
    except:
        print("CSVファイルに書き込むことができませんでした。")


def DataCvt(Data: dict):
    """
    keyに
    * datg,
    * feature_names,
    * target,
    * target_names
    をもつ辞書を引数として、その値から新しい配列を作る関数

    Param
    -----
        Data (dict):
            上記のkeyを持つ辞書。
    
    Return
    ------
        NewData (dict):
    """

    target_name_ = []
    # データが保存されたtupleで渡されたとき
    if data:
        feature = pd.DataFrame(data.data, columns=data.feature_names)

    # データセット内に目的変数名が明記されている場合
    if "target_names" in data:
        # target の列数と target_name の配列の大きさが同じ場合
        if type(data.target_names) is list:
            target_name_ = TargetArr_NamesList(data.target, data.target_names)
        # target の列数と target_name の配列の大きさが異なる場合
        elif type(data.target_names) is np.ndarray:
            target_name_ = TargetArr_NamesArr(data.target, data.target_names)
    # データセット内に目的変数名が明記されていない場合
    # 例えば、住宅価格の配列のみがあるような場合、
    # target = [22.4, 16.5, 19.8, ...]
    else:
        target_name_ = pd.DataFrame(data.target)

    # target_name_を特徴配列の最後尾に合体させる。
    NewData = feature.join(target_name_)

    return NewData


def TargetArr_NamesArr(target_data: np.ndarray, target_names: np.ndarray):
    """
    target の行列が '0, 1, 2'などの整数なら、
    target_name 配列からその値に対応する列数の値を、
    Data のラベルとする。
                
    例えば、
    target = [0, 1, 0, 2, ...]
    target_name = [Apple, Orange, Tomato]
                
    target_name_ = [Apple, Orange, Apple, Tomato, ...]
    """
    target_name_ = []

    if target_data.max() + 1 == target_names.size:
        for i, val in enumerate(data.target):
            target_name = data.target_names[val]
            target_name_.append(target_name)

    target_name_ = pd.DataFrame({"Target_names": target_name_})

    return target_name_


def TargetArr_NamesList(target_data: np.ndarray, target_names: list):
    """
    target の最初の列に target_name の配列を合体させる。
    
    例えば、
    target = [[0, 0, 0],
              [1, 1, 1],
              [2, 2, 2]]
    target_name = [Apple, Orange, Tomato]

    target_name_ = [[Apple, Orange, Tomato],
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2]]
    """
    try:
        if target_data.shape[1] == len(target_names):
            target_name_ = pd.DataFrame(data.target, columns=data.target_names)

    except AttributeError as Att_err:
        print("AttributeError:", Att_err)

    return target_name_


if __name__ == "__main__":
    import os

    path = os.path.dirname(os.path.abspath(__file__))
    # print(path)
    name = "iris"
    # name = "boston"
    # name = "diabetes"
    # name = "digits"
    # name = "linnerrud"
    # name = "wine"
    # name = "breast_cancer"
    path = path + os.sep + name + ".csv"
    data = SKLearnDataSetsLoad(name)

    NewData = DataCvt(data)
    # print(df.tail())
    # print(data.target_names)
    # df = pd.DataFrame(data.data, columns=data.feature_names)
    # target = pd.DataDrame
    CreateDataSetCSV(NewData, path)
