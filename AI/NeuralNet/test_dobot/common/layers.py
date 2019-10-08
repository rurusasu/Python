#coding: utf-8

import sys, os
sys.path.append(os.getcwd())
#from collections import OrderedDict
from common.functions import _CallFunction
import numpy as np


#乗算レイヤ
class mulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # xとyをひっくり返す
        dy = dout * self.x

        return dx, dy


#加算レイヤ
class addLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

#活性化関数のレイヤ
#ReLUレイヤ
class relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)  # xが0以下ならFalseを返す(0より大きければTrue)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        delta = dout

        return delta


#Sigmoidレイヤ
class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        delta = dout * (1.0 - self.out) * self.out

        return delta


#恒等関数レイヤ
class linear:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return self.x

    def backward(self, dout):
        delta = 1 * dout

        return delta

#Affineレイヤ
#アフィン変換を行うレイヤ(重み付き信号の総和を計算する)
class affine:
    def __init__(self, W, b, lr=0.01):
        # パラメータの設定
        self.W = W
        self.B = b
        self.dW = None  # 重みの微分
        self.dB = None  # バイアスの微分
        self.lr = lr    # gradient descentの学習率
        
        self.x = None
        self.original_x_shape = None

    def forward(self, x):
        #テンソル対応
        self.original_x_shape = x.shape  # 元の形を記憶させる
        x = x.reshape(x.shape[0], -1)  # 奥行き方向の幅を固定しつつ、行列の大きさを変更
        self.x = x

        out = self.x.dot(self.W) + self.B

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.dB = np.sum(dout, axis=0)

        #self.W -= self.lr * self.dW
        self.B -= self.lr * self.dB

        dx = dx.reshape(self.original_x_shape)  # 逆伝播を入力信号の形に戻す
        return dx


class InputLayer:
    """
    fit時に入力データの行数を整列する関数

    """
    def __init__(self, input_shape):
        if (type(input_shape) == int):
            self.units = input_shape
        elif (type(input_shape) == tuple):
            self.units = input_shape[0]
        else:
            print('error InputLayer!')

    def _initParams(self, rear_node):
        self.units = (self.units, rear_node)

    def forward(self, data):
        return data
        #batch_mask = np.random.choice(self.units[0], self.units[1], replace=False)


class Dense:
    def __init__(self, Units, activation='relu', weight_initializer='he', bias_initializer='zeros'):
        if (Units != None and type(Units) == int):
            self.units = Units
        self.activation = activation
        self.initializer  = {} 
        self.initializer['W'] = weight_initializer
        self.initializer['b'] = bias_initializer
        self.params = {}   
        self.function = {}

        
    def _initParams(self, rear_node):
        # 重みの初期化
        self.__init_weight(self.initializer['W'], rear_node)
        self.__init_bias(self.initializer['b'], rear_node)

    
    def __init_weight(self, weight_initializer, rear_node):
        """
        重みの初期設定

        Parameters
        ----------
        weight_initializer : 重みの標準偏差を指定
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        rear_node : 次のnodeの行数
        """
        if str(weight_initializer).lower() in ('relu', 'he'):
            weight_initializer = 'he_nomal'
        elif str(weight_initializer).lower() in ('sigmoid', 'xavier'):
            weight_initializer = 'glorot_uniform'
        method = _CallFunction('weight', weight_initializer)
        scale = method(self.units)
        self.params['W'] = scale * np.random.randn(self.units, rear_node)

   
    def __init_bias(self, bias_initializer, rear_node):
        """
        重みの初期設定

        Parameters
        ----------
        bias_initializer : biasを指定
        rear_node        : 次のnodeの行数
        """
        method = _CallFunction('bias', bias_initializer)
        self.params['b'] = method(rear_node)


    def setFunc(self, lr):
        """
        ユニット内部関数をセットする

        Parameters
        ----------
        lr : float
            学習率
        """
        #レイヤの設定
        self.function['Affine'] = globals()['affine'](self.params['W'], self.params['b'])  # アフィン変換を行うレイヤをセット
        self.function['Activation'] = globals()[self.activation]()
    
    
    def _optimizer(self, optimizer='sgd', loss=1, lr=0.01):
        method = _CallFunction('optimizer', optimizer)
        print(self.params.keys())
        #method.update(self.params, loss)


    def forward(self, x):
        for layer in self.function.values():
            x = layer.forward(x)
        return x


if __name__ == "__main__":
    #print(InputLayer(2))
    #print(InputLayer((2, 3)))
    
    dense = Dense(50)
    #Dense(weight_initializer='relu')
    dense._initParams(50)
    dense._optimizer()
