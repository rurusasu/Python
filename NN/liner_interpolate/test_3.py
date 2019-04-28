import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.layers_2 import *
from collections import OrderedDict


class Sequential:
    #sequential = []
    values = []
    history = {}
    count = 0
    Dictionaly = {}

    def __init__(self):
        #pass
        self.sequential = []


    def add(self, layer_name):
        self.sequential.append(layer_name)
        #Sequential.sequential.append(layer_name) #リストにレイヤの名前を代入
        #LayerKey = ('Layer' + '%s' %(Sequential.count+1))
        #Sequential.count += 1
        #LayerDict = dict(LayerKey, layer_name)
        #Sequential.sequential.update(Sequential.sequential, LayerDict)

    def compile(self, loss):
        #for i in range(len(Sequential.sequential)):
            #LayerKey = ('Layer' + '%s' %(i+1))
            #values = dict(LayerKey, Sequential.sequential[i])
            #Sequential.Dictionaly.update(values)
            #self.Layers = self.Predict()
        x = 0
        for layers in self.sequential:
            x = layers.unit(x)

        self.LastLayer = globals()[loss]


    def fit(self, train_data, test_data, batch_size, epochs):
        x = train_data
        t = test_data
        TrainRow_size = x.shape[0] # train_dataの行数を取得 返り値：整数
        TrainCol_size = x.shape[1] # train_dataの列数を取得 返り値：整数

        for i in range(epochs):
            for batch in range(batch_size):
                batch_mask = np.random.choice(TrainRow_size, batch_size, replace = False) #行数からランダムに値を抽出 replace(重複)
                x_batch = x[batch_mask, 0:TrainCol_size] #全データからbatch_size分データを抽出
                t_batch = t[batch_mask]

                for layers in self.sequential:
                    x_batch = layers.forward(x_batch)
                
                Sequential.values.append(self.LastLayer.function(x_batch, t_batch))

                x = np.delete(x, batch_mask, 0) #全データから使用したbatchデータを削除
                t = np.delete(t, batch_mask)


            if train_data.shape[0] % batch_size == 0:
                layer.reverse()

        Sequential.history['loss'] = Sequential.values

    #def evaluate(self, x_test, y_test):


    ########################################
    ########      内部関数       ###########
    ########################################
    def Predict(self):
        y = None
        for layer in Sequential.sequential.values():
            y = layer.unit(y)
            Sequential.values.append(y)

        Layers = dict(zip(Sequential.sequential, Sequential.values))
        return Layers

    def Loss(self, loss):
        self.y = Predict()




class InputLayer:
    def __init__(self, input_shape):
        #self.input = {}
        if len(input_shape) == 1: #もし、入力数が配列で指定されたとき
            self.UnitsCol = 1     #配列をn行1列として考える

        elif len(input_shape) == 2:
            self.UnitsCol = input_shape[1]

    def unit(self, y):
        return self.UnitsCol

    def forward(self, input_data):
        return input_data


class Dense:
    def __init__(self, units, activation):
        self.dense = OrderedDict()                       #関数の辞書
        self.params = {}                                 #ユニット内での計算に必要なパラメータの辞書
        #ある層の情報
        #self.dense['Activation'] = globals()[activation] #活性化関数名
        self.params['Units'] = units                     #ユニットの数
        self.params['Weight'] = None                     #重み
        self.params['Bias'] = None                       #閾値
        self.activation = activation

    def InitParams(self, input_size):
        K = 2
        #初期値の計算
        self.params['Weight'] = K*(np.ones((input_size, self.params['Units']))*0.5 - np.random.rand(input_size, self.params['Units'])) #重み
        self.params['Bias']   = np.zeros(self.params['Units'])                                                                         #閾値

    def unit(self, input_size):
        #初期値を設定
        self.InitParams(input_size)
        #活性化関数を設定
        #self._class = globals()[self.dense['Activation']]
        self.dense['Affine'] = globals()['affine'](self.params['Weight'], self.params['Bias']) #アフィン変換を行うレイヤをセット
        self.dense['Activation'] = globals()[self.activation]

        return self.params['Units']

    def forward(self, input_data):
        for i in range(len(self.dense)):
            x = self.dense['Affine'].forward(input_data)
            x = self.dense['Activation'].forward(x)






data = np.loadtxt(
    "save_data.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )




#読み込んだデータを学習用にコピーする
data_1 = data_copy(data)

#標準化
data_ = data_std(data_1)

#正規化
data_ = data_nom(data_)


x_train = data_[:, 0:2] #学習データをセット
y_train = data_[:, 2]   #正解データをセット

module = Sequential()
module.add(InputLayer(input_shape = (20,2)))
module.add(Dense(1, activation = 'liner'))
module.compile('mean_squared_error')

#学習
epochs = 20
batch_size = 20
history = module.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)