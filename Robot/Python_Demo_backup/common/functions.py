#coding: utf-8

import sys, os
sys.path.append(os.getcwd())

import numpy as np
import importlib
import inspect

# ステップ関数
def StepFunction(x):
    return np.array(x > 0, dtype=np.int)

# シグモイド関数
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU関数(ランプ関数)
def Relu(x):
    return np.maximum(0, x)

#---------------------------------
#出力層
#---------------------------------
# 恒等関数(回帰問題で使用)
def LinerFunction(x):
    return x

# tanh関数
def tanh(x):
    return np.tanh(x)

# logistic関数
def logistic(x):
    y = 1 / (1 + np.exp(-x))
    return y

# ソフトマックス関数(分類問題で使用)
def Softmax(x):
    if x.ndim == 2:
        x = x.T
        y = x - np.max(x, axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

#--------------------------------------------
#損失関数
#--------------------------------------------
# 2乗和誤差
def mean_squared_error2(y, t):
    return 0.5 * np.sum((y-t)**2, axis=1, keepdims=True)


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def MeanSquaredError(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    out = 0.5 * np.sum((y-t)**2) / batch_size

    return out

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    #教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


#################################
####    新しく機能を追加    #####
#################################
# データコピー
def data_copy(data):
    dim = data.ndim  # 受け取ったdataの次元を確認

    if dim == 1:  # dataが1次元(配列)のとき
        return data
    else:  # dataが2次元(行列)のとき
        data_copy = np.empty_like(data)  # dataと同じ大きさの空の行列を作成
        col = data.shape[1]  # dataの列数を取得

        for i in range(col):
            data_copy[:, i] = data[:, i]  # 対応する列に値をコピー

        return data_copy

# 標準化
def data_std(data):
    dim = data.ndim  # 受け取ったdataの次元を確認

    if dim == 1:  # dataが1次元(配列)のとき
        return (data - data.mean()) / data.std()
    else:  # dataが2次元(行列)のとき
        data_std = np.empty_like(data)  # dataと同じ大きさの空の行列を作成
        row = data.shape[0]  # dataの行数を取得
        col = data.shape[1]  # dataの列数を取得

        #for i in range(row):
            #data_std[i, j] =(data[i, j] - data[i, j].mean()) / data[i, j].std() #対応する列を標準化

        for i in range(col):
            data_std[:, i] = (data[:, i] - data[:, i].mean()) 
                                                        / data[:, i].std()

        return data_std


# 正規化
def data_nom(data):
    dim = data.ndim  # 受け取ったdataの次元を確認

    if dim == 1:  # dataが1次元(配列)のとき
        return data_1[:, 0] / max(abs(data_1[:, 0]))
    else:  # dataが2次元(行列)のとき
        data_nom = np.empty_like(data)  # dataと同じ大きさの空の行列を作成
        row = data.shape[0]  # dataの行数を取得
        col = data.shape[1]  # dataの列数を取得

        #for i in range(row):
        #for j in range(col):
        #data_nom[i, j] = data[i, j] / max(abs(data[i, j])) #対応する列を正規化

        for i in range(col):
            data_nom[:, i] = data[:, i] / max(abs(data[:, i]))

        return data_nom


def _Call(function_name):
    return importlib.import_module(function_name)


def _CallFunction(module, function):
    """
    Call function with module

    Parameters
    ----------
    module : str
        呼び出したいmodule名
    function : str
        module内の関数名
    
    Return
    ------
    method : method
        実行可能な関数
    """
    m = _Call(module)  # モジュール呼び出し
    if (function in dir(m)):  # 関数がモジュール内にあるか確認
        return getattr(m, function)  # 関数呼び出し
    else:
        print('functionが存在しません！')


def _CallClass(mod_name, cls_name):
    # from optimizer import sgd
    cls = __import__(mod_name, fromlist=[cls_name])
    instance = getattr(cls, cls_name)
    return instance


if __name__ == "__main__":
    class_def = _CallClass('optimizer', 'sgd')
    obj = class_def()
    obj.update()
