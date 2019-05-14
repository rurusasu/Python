#coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from collections import OrderedDict

#乗算レイヤ
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

    def backward(self, dout):
        dx = dout * self.y #xとyをひっくり返す
        dy = dout * self.x

        return dx, dy


#加算レイヤ
class AddLayer:
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
    
    def forward(self, x, counter):
        self.mask = (x <= 0) #xが0以下ならFalseを返す(0より大きければTrue)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


#Sigmoidレイヤ
class sigmoid:
    def __init__ (self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        delta = dout * (1.0 - self.out) * self.out

        return dx


#恒等関数レイヤ
class liner:
    def __init__(self):
        self.x = None 

    def forward(self, x, counter):
        self.x = x
        return self.x

    def backward(self, dout):
        delta = 1 * dout

        return delta


#Affineレイヤ
#アフィン変換を行うレイヤ(重み付き信号の総和を計算する)
class affine:
    def __init__(self, W, b):
        self.W = W
        self.B = b

        self.x = None
        self.original_x_shape = None
        #重み・バイアスパラメータの微分
        self.dW = None
        self.dB = None

    def forward(self, x, counter):
        #テンソル対応
        self.original_x_shape = x.shape #元の形を記憶させる
        x = x.reshape(x.shape[0], -1)   #奥行き方向の幅を固定しつつ、行列の大きさを変更
        self.x = x
        
        out = np.dot(self.x, self.W) + self.B

        #printの設定
        print('第%d層 - AffineLayer - Weight%d, %d' %(counter, counter-1, counter))
        print(self.W)
        print('第%d層 - AffineLayer - Bias%d' %(counter, counter))
        print(self.B)

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.W = np.dot(self.x.T, dout)
        self.B = np.sum(dout, axis = 0)

        dx = dx.reshape(*self.original_x_shape) #逆伝播を入力信号の形に戻す
        return dx


#出力層
#2乗和誤差レイヤ
class mean_squared_error:
    def __init__(self):
        self.loss = None #損失
        self.y    = None #linerの出力
        self.t    = None #教師データ

    def forward(self, x, t):
        self.y = x
        self.t = t
        if t.shape != x.shape:
            self.y = self.t.reshape(self.y.size, 1)
            self.t = self.y.reshape(self.t.size, 1)
        self.loss = MeanSquaredError(self.y, self.t)

        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        if self.y.size == self.t.size:
            dx = (self.y - self.t) / batch_size

        return dx


#####損失関数#####
class LinerWithLoss:
    def __init__(self):
        self.loss = None #損失
        self.y    = None #linerの出力
        self.t    = None #教師データ

    def forward(self, x, t):
        self.y = x
        self.t = t
        if t.shape != x.shape:
            self.y = self.y.reshape(self.y.size, 1)
            self.t = self.t.reshape(self.t.size, 1)
        self.loss = MeanSquaredError(self.y, self.t)

        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        if self.y.size == self.t.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


#Softmax & 交差エントロピー誤差を含めた計算を行うレイヤ
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #損失
        self.y = None    #softmaxの出力
        self.t = None    #教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
    
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


###################################
####    新しくレイヤを追加    #####
###################################
#入力レイヤ
class InputLayer:
    def __init__(self, input_shape):
        #self.Input_Row_Size = input_shape
        self.Input_Col_Size = None
        self.input          = None
        #
        #if len(input_shape) == 1:   #もし、入力数が配列で指定されたとき
            #self.input = 1
            #pass
        #elif len(input_shape) == 2:
            #self.input = input_shape[1]


    def unit(self, Data_Col_Size, counter):
        print('第%d層 - InputLayer' %counter)
        self.Input_Col_Size = Data_Col_Size
    
        return self.Input_Col_Size

    def forward(self, input_data):
        out = np.reshape(input_data, [-1, self.Input_Col_Size])

        return out


class InputLayer2:
    def __init__(self, input_shape):
        self.input_data = None
        if len(input_shape) == 1:   #もし、入力数が配列で指定されたとき
            self.input = 1
        elif len(input_shape) == 2:
            self.input = input_shape[1]

    def unit(self, y, counter):
        print('第%d層 - InputLayer' %counter)

        return self.input

    def forward(self, input_data):
        self.input_data = input_data
        return self.input_data

    def backward(self, dout):
        pass


#全結合レイヤ
class Dense:
    def __init__(self,  Units, activation, weight_initializer='glorot_uniform', bias_initializer='zeros'):
        self.dense = OrderedDict()         #関数の辞書
        self.RevDense = None               #関数の辞書の反転(逆伝播で使用)
        self.activation = activation       #活性化関数名

        self.params = {}               #ユニット内での計算に必要なパラメータの辞書
        self.params['Units']  = Units  #ユニットの数
        self.params['Weight'] = None   #重み
        self.params['Bias']   = None   #閾値

        #####   初期値の設定   #####
        self.initialisation   = InitParams()
        self.init_weight      = weight_initializer
        self.init_bias        = bias_initializer

        self.counter          = None
        

    def initparams(self, BefLayer_Size, Unit_size):
        #K = 2
        #初期値の計算
        #weight =  K*(np.ones((input_size, self.params['Units']))*0.5 - np.random.rand(input_size, self.params['Units'])) #重み
        weight = InitParams.glorot_uniform(input_size=BefLayer_Size, hidden_size=Unit_size)
        bias   = np.zeros(self.params['Units'])                                                                         #閾値
                
        return weight, bias

    def unit(self, BefLayer_Size, counter):
        #初期値を設定
        self.params['Weight'] = self.initialisation.glorot_uniform(BefLayer_Size, self.params['Units'])
        self.params['Bias']   = np.zeros(self.params['Units'])  
        #活性化関数を設定
        self.dense['Affine'] = globals()['affine'](self.params['Weight'], self.params['Bias']) #アフィン変換を行うレイヤをセット
        self.dense['Activation'] = globals()[self.activation]()                                #活性化関数のレイヤをセット

        self.counter = counter
        print('第%d層 - AffineLayer' %self.counter)
        print('第%d層 - Activation %s' %(self.counter, self.activation))

        return self.params['Units']


    def forward(self, input_data):
        x = input_data
        for layer in self.dense.values():
            x = layer.forward(x, self.counter)

        return x


    def backward(self, dout):
        x = dout
        self.RevDense = list(self.dense.values()) #OrederedDictを使う場合、内部の値を入れ替える際はlistにする必要がある。
        self.RevDense.reverse()
        for RevLayer in self.RevDense:
            x = RevLayer.backward(x)

        return x


#重みの初期値計算
class InitParams:
    def __init__(self):
        pass

    #重みの初期値
    #Xavierの一様分布
    def glorot_uniform(self, input_size, hidden_size):
        weight = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)

        return weight