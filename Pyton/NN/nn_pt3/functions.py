#coding: utf-8
import numpy as np


#ステップ関数
def StepFunction(x):
    return np.array(x > 0, dtype=np.int)


#シグモイド関数
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

#ReLU関数(ランプ関数)
def Relu(x):
    return np.maximum(0, x)


#出力層
#恒等関数(回帰問題で使用)
def LinerFunction(x):
    return x


#ソフトマックス関数(分類問題で使用)
def Softmax(x):
    if x.ndim == 2:
        x = x.T
        y = x - np.max(x, axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


#損失関数
#2乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2, axis=1, keepdims=True)


#交差エントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    #教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis = 1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


#################################
####    新しく機能を追加    #####
#################################

#データコピー
def data_copy(data):
    dim = data.ndim                      #受け取ったdataの次元を確認

    if dim == 1 :                        #dataが1次元(配列)のとき
        return data
    else :                               #dataが2次元(行列)のとき
        data_copy = np.empty_like(data)  #dataと同じ大きさの空の行列を作成
        col = data.shape[1]              #dataの列数を取得

        for i in range(col):
            data_copy[:, i] = data[:, i] #対応する列に値をコピー
        
        return data_copy

#標準化
def data_std(data):
    dim = data.ndim #受け取ったdataの次元を確認

    if dim == 1 :   #dataが1次元(配列)のとき
        return (data - data.mean()) / data.std()
    else :                               #dataが2次元(行列)のとき
        data_std = np.empty_like(data)   #dataと同じ大きさの空の行列を作成
        row = data.shape[0]              #dataの行数を取得
        col = data.shape[1]              #dataの列数を取得

        #for i in range(row):
            #for j in range(col):
                #data_std[i, j] =(data[i, j] - data[i, j].mean()) / data[i, j].std() #対応する列を標準化
        
        for i in range(col):
            data_std[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].std()

        return data_std


#正規化
def data_nom(data):
    dim = data.ndim #受け取ったdataの次元を確認

    if dim == 1 :   #dataが1次元(配列)のとき
        return data_1[:, 0] / max(abs(data_1[:, 0]))
    else :                               #dataが2次元(行列)のとき
        data_nom = np.empty_like(data)   #dataと同じ大きさの空の行列を作成
        row = data.shape[0]              #dataの行数を取得
        col = data.shape[1]              #dataの列数を取得

        #for i in range(row):
            #for j in range(col):
                #data_nom[i, j] = data[i, j] / max(abs(data[i, j])) #対応する列を正規化
        
        for i in range(col):
            data_nom[:, i] = data[:, i] / max(abs(data[:, i]))

        return data_nom


# バッチデータを選択する
class DataChoice:
    def __init__(self):
        self.BatchMask = 0


    def RandomChoice(self, choice_data, choice_length, choice_size):
        self.BatchMask = np.random.choice(choice_length, choice_size, replace = False) #行数からbatch_sizeだけランダムに値を抽出 replace(重複)
        
        #全データから
        if (len(choice_data.shape) == 1):
            choice_data = choice_data[self.BatchMask] #batch_size分データを抽出
        else:
            ColSize = choice_data.shape[1]
            choice_data = choice_data[self.BatchMask, 0:ColSize]

        return choice_data


    def DataDelete(self, delete_data, delete_size = None):
        # もし、特定の個数データをランダムに削除したい場合
        if delete_size != None:
            self.BatchMask = np.random.choice(delete_data.size, delete_size, replace = False)

        if (len(delete_data.shape) == 1):
            delete_data = np.delete(delete_data, self.BatchMask)
        
        else:
            delete_data = np.delete(delete_data, self.BatchMask, 0)

        return delete_data