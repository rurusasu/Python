import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.layers_2p import *
from collections import OrderedDict

class output: pass

class Sequential:

    def __init__(self):
        self.sequential = []
        self.Output = output()
        self.Output.history = {}

        self.Output.history['loss']    = []
        self.Output.history['acc']     = []
        self.Output.history['val_acc'] = []

    def add(self, layer_name):
        #リストにレイヤの名前を代入
        self.sequential.append(layer_name)

    def compile(self, loss):
        x = 0
        for layers in self.sequential:
            x = layers.unit(x)

        self.LastLayer = globals()[loss]()


    def fit(self, x_train, t_train, batch_size, epochs, validation_data):
        x = x_train
        t = t_train
        #loss = []
        TrainRow_size = x.shape[0] # train_dataの行数を取得 返り値：整数
        TrainCol_size = x.shape[1] # train_dataの列数を取得 返り値：整数

        for i in range(epochs):
            batch_mask = np.random.choice(TrainRow_size, batch_size, replace = False) #行数からbatch_sizeだけランダムに値を抽出 replace(重複)
            x_batch = x[batch_mask, 0:TrainCol_size] #全データからbatch_size分データを抽出
            t_batch = t[batch_mask]

                
            #推論を行う
            output = self.Predict(x_batch)
                
            #誤差を保存する
            #loss.append(self.LastLayer.forward(output, t_batch))
            loss = self.LastLayer.forward(output, t_batch)

            #全データから使用したbatchデータを削除。削除された分データは詰められる
            x = np.delete(x, batch_mask, 0)
            t = np.delete(t, batch_mask)
            TrainRow_size = TrainRow_size - batch_size

            #1エポックが終了すると重みと閾値を更新する
<<<<<<< HEAD
            AveLoss = np.sum(loss, axis = 1) / batch_size  #誤差の平均値(誤差の合計 ÷ バッチサイズ)を計算
            AveLoss = np.sum(AveLoss, axis = 0) / AveLoss.size
            self.Output.history['loss'].append(AveLoss)    #lossリストに値を格納
            dout = AveLoss

            loss = [] #lossを再初期化
=======
            #AveLoss = np.sum(loss, axis = 1) / batch_size  #誤差の平均値(誤差の合計 ÷ バッチサイズ)を計算
            #self.Output.history['loss'].append(AveLoss)
            self.Output.history['loss'].append(loss)
            #dout = AveLoss
            #loss = [] #lossを再初期化
>>>>>>> 2874cc5e857b7145a93b67491616fc22180a4cab

            #逆伝播を行うためにレイヤを反転
            self.sequential.reverse()

            #逆伝搬および重みの更新
            dout = 1
            dout = self.LastLayer.backward(dout)
            for layer in self.sequential:
                dout = layer.backward(dout)
            
            self.sequential.reverse()

            #正解率を計算
            train_acc = self.accuracy(x_batch, t_batch)
            test_acc  = self.accuracy(validation_data[0], validation_data[1])
            self.Output.history['acc'].append(train_acc)
            self.Output.history['val_acc'].append(test_acc)

        return self.Output


    ########################################
    ########      内部関数       ###########
    ########################################
    def Predict(self, x_batch):
        for layers in self.sequential:
            x_batch = layers.forward(x_batch)

        return x_batch

    def accuracy(self, x, t):
        y = self.Predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


class InputLayer:
    def __init__(self, input_shape):
        self.input_data = None
        if len(input_shape) == 1:   #もし、入力数が配列で指定されたとき
            self.input = 1
        elif len(input_shape) == 2:
            self.input = input_shape[1]

    def unit(self, y):
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
        self.activation = activation                     #活性化関数名

        self.params = {}                                 #ユニット内での計算に必要なパラメータの辞書
        self.params['Units'] = units                     #ユニットの数
        self.params['Weight'] = None                     #重み
        self.params['Bias'] = None                       #閾値

    def InitParams(self, input_size):
        K = 2
        #初期値の計算
        weight =  K*(np.ones((input_size, self.params['Units']))*0.5 - np.random.rand(input_size, self.params['Units'])) #重み
        bias   =  np.zeros(self.params['Units'])                                                                         #閾値
        
        return weight, bias

    def unit(self, input_size):
        #初期値を設定
        self.params['Weight'], self.params['Bias'] = self.InitParams(input_size)
        #活性化関数を設定
        self.dense['Affine'] = globals()['affine'](self.params['Weight'], self.params['Bias']) #アフィン変換を行うレイヤをセット
        self.dense['Activation'] = globals()[self.activation]()                                #活性化関数のレイヤをセット

        return self.params['Units']


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
t_train = data_[:, 2]   #正解データをセット

#テストデータのセット
x_test  = test_[:, 0:2] #入力データをセット
t_test  = test_[:, 2]   #正解データをセット


module = Sequential()
module.add(InputLayer(input_shape = (20,2)))
module.add(Dense(50, activation = 'relu'))
module.add(Dense(1,  activation = 'relu'))
module.compile('LinerWithLoss')

#学習
epochs = 2
batch_size = 2
history = module.fit(x_train, t_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, t_test))

#lossグラフ
loss    = history.history['loss']
val_acc = history.history['val_acc']

nb_epoch = len(loss)
plt.plot(range(nb_epoch), loss,     marker = '.', label = 'acc')
plt.plot(range(nb_epoch), val_acc, marker = '.', label = 'val_acc')
plt.legend(loc = 'best', fontsize = 10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()