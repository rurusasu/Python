#coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *

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
    
    def forward(self, x):
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
        dx = dout * (1.0 - self.out) * self.out

        return dx


#Affineレイヤ
#アフィン変換を行うレイヤ(重み付き信号の総和を計算する)
class affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        #重み・バイアスパラメータの微分
        self.dW = None
        self.dB = None

    def forward(self, x):
        #テンソル対応
        self.original_x_shape = x.shape #元の形を記憶させる
        x = x.reshape(x.shape[0], -1)   #奥行き方向の幅を固定しつつ、行列の大きさを変更
        self.x = x
        
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.dB = np.sum(dout, axis = 0)

        dx = dx.reshape(*self.original_x_shape) #逆伝播を入力信号の形に戻す
        return dx


#出力層
#恒等関数レイヤ
class liner:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return self.x

    def backward(self, dout):
        dx = dout
        return dx


#2乗和誤差レイヤ
class mean_squared_error:
    def __init__(self):
        self.loss = None
        self.t = None
        self.original_x_shape = None

    def forward(self, x, t):
        #self.x = x
        self.original_x_shape = x.shape #元の形を記憶させる
        self.t = t.reshape(x.shape[0], -1)
        #self.t.reshape(x.shape[0], 1)
        self.loss = MeanSquaredError(x, self.t)
        self.loss = self.loss.reshape(*self.original_x_shape)

        return self.loss

    def backward(self, dout = 1):
        dx = dout * (self.loss - self.t)

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
        batch_size = self.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

