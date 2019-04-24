import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.layers import *
from collections import OrderedDict

A = np.arange(-100, 100, 1)
B = np.arange(1, 5, 1)
print(A[B])

A_1 = A.reshape(-1, 5)
print(A_1)
print(A_1[B,0:2])

x = np.arange(-100, 100, 1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) -0.5

y = sigmoid(x)


class Sequential:
    sequential = []
    values = []

    def __init__(self):
        pass


    def add(self, layer_name):
        Sequential.sequential.append(layer_name)


    def compile(self, loss):
        for layer in Sequential.sequential:
            self.y = layer.unit(self.y)
            Sequential.value.append(self.y)

        self.LastLayer = globals()[loss]

        self.Layers = dict(zip(Sequential.sequential, Sequential.values))

    def fit(train_data, test_data, batch_size, epoche):
        for i in range(epotch):
            for j in self.Layers:
                x = layer.forward(x)

                return x


class test:
    def init(self, input = 3, units = 2, layer = 2):
        K = 2
        WeightKeys = []
        WeightValues = []
        BiasKeys = []
        BiasValues = []
        a = 'W'
        b = 'b'

        for i in range(layer):
            #層の重みを辞書形式で作成する
            WeightKey = ('W' + '%s' %(i+1)) #辞書の鍵を生成する
            WeightValue = K*(np.ones((input, units))*0.5 - np.random.rand(input, units)) #鍵に対応する値を生成する
            WeightKeys.append(WeightKey) #それぞれの鍵を配列にする
            WeightValues.append(WeightValue) #対応する値を配列にする
            self.WeightParams = dict(zip(WeightKeys, WeightValues)) #辞書としてまとめる

            #層の閾値を辞書形式で作成する
            BiasKey = ('B' + '%s' %(i+1)) #辞書の鍵を生成する
            BiasValue = np.zeros(units) #鍵に対応する値を生成する
            BiasKeys.append(BiasKey) #それぞれの鍵を配列にする
            BiasValues.append(BiasValue) #対応する値を配列にする
            self.BiasParams = dict(zip(BiasKeys, BiasValues)) #辞書としてまとめる

        #print(WeightParams)
        #print(BiasParams)

    def Layer(self, units, Activation):
        self.init()
        b_class = globals()[Activation]
        b = b_class(self.WeightParams, self.BiasParams)
    
        print(b)


class InputLayer:
    def __init__(self, input_shape):
        for num in input_shape:
            self.Units = num

    def unit(self, y):
        return self.Units

class Dense:
    def __init__(self, units, activation):
        #ある層の情報
        self.Units = units           #ユニットの数
        self.Activation = activation #活性化関数名
        self.W = None                #重み
        self.B = None                #閾値

    def InitParams(self, input_size):
        K = 2
        #初期値の計算
        self.W = K*(np.ones((input_size, self.Units))*0.5 - np.random.rand(input_size, self.Units)) #重み
        self.B = np.zeros(self.Units)                                                               #閾値

    def unit(self, input_size):
        #初期値を設定
        self.InitParams(input_size)
        #活性化関数を設定
        self._class = globals()[self.Activation]

        return self.Units




#plt.plot(x, y)
#plt.xlabel("x")
#plt.ylabel("y")
#plt.show()

W1 = np.ones((3, 5)) #重みは全結合の場合
b1 = np.ones(5) #閾値は重みの列数と同じ数だけ必要
W2 = np.ones((5, 1))
b2 = np.ones(1)
X = np.array([1.0, 2.0, 3.0])
Y1 = Affine(W1, b1)
Y2 = Affine(W2, b2)
Z1 = Y1.forward(X)
Z2 = Y2.forward(Z1)
print(Z1, Z2)

#dout = np.array((1, 2))
#print(dout)
#Z3 = Y1.backward(dout)
#print(Z3)

y = test()
y.Layer(50, 'Affine')

module = Sequential()
module.add(InputLayer(input_shape = (784,)))
module.add(Dense(10, activation = 'softmax'))
module.compile('categorical_crossentropy')

#学習
epoches = 20
batch_size = 20
history = model.fit()