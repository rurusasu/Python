import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.layers_2p import *
from collections import OrderedDict


class Sequential:
    #sequential = []
    loss = []
    history = []
    count = 0
    Layers = {}

    def __init__(self):
        #pass
        #self.sequential = OrderedDict()
        #self.sequential = {'input':None}
        self.sequential = []

    def add(self, layer_name):
        self.sequential.append(layer_name)
        #self.sequential.update(layer_name)
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
        #for layer in self.sequential:
            #Sequential.Layers.update(layer)

        x = 0
        for layers in self.sequential:
            x = layers.unit(x)
            #LayerParams = x.values()

        self.LastLayer = globals()[loss]()


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

                
                #推論を行う
                for layers in self.sequential:
                    x_batch = layers.forward(x_batch)
                
                #誤差を保存する
                Sequential.loss.append(self.LastLayer.forward(x_batch, t_batch))

                #全データから使用したbatchデータを削除。削除された分データは詰められる
                x = np.delete(x, batch_mask, 0)
                t = np.delete(t, batch_mask)
                TrainRow_size = TrainRow_size - batch_size

            #1エポックが終了すると重みと閾値を更新する
            AveLoss = np.sum(Sequential.loss, axis = 1) / batch_size  #誤差の平均値(誤差の合計 ÷ バッチサイズ)を計算
            Sequential.history.append(AveLoss)
            dout = AveLoss

            Sequential.loss = []  #lossを再初期化

            #layers = list(self.LastLayer.values())
            #逆伝播を行うためにレイヤを反転
            self.sequential.reverse()
            #ReLayers.reverse()
            #layers.reverse()
            #逆伝搬および重みの更新
            for layer in self.sequential:
                dout = layer.backward(dout)
            
            self.sequential.reverse()
            #重みの更新
            #for i in range(len(self.sequential) -1):
                #self.sequential[i].params['Weight'], self.sequential[i].params['Bias'] = \
                    #self.sequential[i].dense['Affine'].dW, self.sequential[i].dense['Affine'].dB
            #self.sequential[1].
            #grads['W']
        #Sequential.history['loss'] = Sequential.values

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
        #self.input = OrderedDict()
        #self.input = {}
        self.input_data = None
        if len(input_shape) == 1:   #もし、入力数が配列で指定されたとき
            #self.input['input'] = 1 #配列をn行1列として考える
            self.input = 1
        elif len(input_shape) == 2:
            #self.input['input'] = input_shape[1]
            self.input = input_shape[1]

    def unit(self, y):
        #return self.input.values() #入力データの列数を返す
        return self.input

    def forward(self, input_data):
        self.input_data = input_data
        return self.input_data

    def backward(self, dout):
        pass


class Dense:
    def __init__(self, units, activation):
        self.dense = OrderedDict()                       #関数の辞書
        self.RevDense = None                             #関数の辞書の反転(逆伝播で使用)
        self.params = {}                                 #ユニット内での計算に必要なパラメータの辞書
        #ある層の情報
        #self.dense['Activation'] = globals()[activation] #活性化関数名
        self.params['Units'] = units                     #ユニットの数
        self.params['Weight'] = None                     #重み
        self.params['Bias'] = None                       #閾値
        self.activation = activation

    def InitParams(self, input_size):
        K = 2
        #input_size = input_unit_size.values()
        #初期値の計算
        #self.params['Weight'] = K*(np.ones((input_size, self.params['Units']))*0.5 - np.random.rand(input_size, self.params['Units'])) #重み
        #self.params['Bias']   = np.zeros(self.params['Units'])                                                                         #閾値
        weight =  K*(np.ones((input_size, self.params['Units']))*0.5 - np.random.rand(input_size, self.params['Units'])) #重み
        bias   =  np.zeros(self.params['Units'])                                                                         #閾値
        
        return weight, bias

    def unit(self, input_size):
        #初期値を設定
        self.params['Weight'], self.params['Bias'] = self.InitParams(input_size)
        #活性化関数を設定
        #self._class = globals()[self.dense['Activation']]
        self.dense['Affine'] = globals()['affine'](self.params['Weight'], self.params['Bias']) #アフィン変換を行うレイヤをセット
        self.dense['Activation'] = globals()[self.activation]()
        #self.dense['Activation'] = globals()['liner']()
        #self.RevDense = self.dense
        #reversed(self.RevDense)

        return self.params['Units']
        #return self.dense

    def forward(self, input_data):
        x = input_data
        for layer in self.dense.values():
            x = layer.forward(x)

        return x

    def backward(self, dout):
        x = dout
        self.RevDense = list(self.dense.values()) #OrederedDictを使う場合、内部の値を入れ替える際はlistにする必要がある。
        self.RevDense.reverse()
        for RevLayer in self.RevDense:
            x = RevLayer.backward(x)

        return x





#訓練データの読み込み
data = np.loadtxt(
    "save_data.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )

#テストデータの読み込み
test = np.loadtxt(
    "test_data.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )



#読み込んだデータを学習用にコピーする
data_1 = data_copy(data)
test_1 = data_copy(test)

#標準化
data_ = data_std(data_1)
test_ = data_std(test_1)

#正規化
data_ = data_nom(data_)
test_ = data_nom(test_)

#訓練データのセット
x_train = data_[:, 0:2] #入力データをセット
y_train = data_[:, 2]   #正解データをセット

#テストデータのセット
x_test  = test_[:, 0:2] #入力データをセット
y_test  = test_[:, 2]   #正解データをセット

module = Sequential()
module.add(InputLayer(input_shape = (20,2)))
module.add(Dense(50, activation = 'liner'))
module.add(Dense(1,  activation = 'liner'))
module.compile('mean_squared_error')

#学習
epochs = 20
batch_size = 2
history = module.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)