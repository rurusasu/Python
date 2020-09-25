import sys

sys.path.append(".")
sys.path.append("..")

import pandas as pd
from sklearn.datasets import *


def SKLearnDataSetsLoad(DataSetName):
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


def DataCvt(Data):
    target_name_ = []
    # データが保存されたtupleで渡されたとき
    if data:
        feature = pd.DataFrame(data.data, columns=data.feature_names)

    # データセット内に目的変数名が明記されている場合
    if "target_names" in data:
        if data.target.shape == data.target_names.shape:
            target = pd.DataFrame(data.target, columns=data.target_names)
        else:
            for i, val in enumerate(data.target):
                target_name = data.target_names[val]
                target_name_.append(target_name)
            target_name_ = pd.DataFrame({'Target_names':target_name_})
            # print(target_name_)
            NewData = feature.join(target_name_)

            #print(NewData)
    # データセット内に目的変数名が明記されていない場合
    else:
        target = pd.DataFrame(data.target)
        # print("target_namesはありません。")

    print(feature)
    print(NewData)

    return 0


def CreateDataSetCSV(df, path):
    try:
        df.to_csv(path)
    except:
        print("CSVファイルに書き込むことができませんでした。")


if __name__ == "__main__":
    import os

    path = os.path.dirname(os.path.abspath(__file__))
    # print(path)
    # name = "iris"
    name = "boston"
    #name = "breast_cancer"
    path = path + os.sep + name + ".csv"
    data = SKLearnDataSetsLoad(name)

    DataCvt(data)
    # print(df.tail())
    # print(data.target_names)
    # df = pd.DataFrame(data.data, columns=data.feature_names)
    # target = pd.DataDrame
    # CreateDataSetCSV(df, path)
