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
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        self.dout = dout
        if self.x.ndim == 1:
            self.x = self.x.reshape(1, self.x.shape[0])
        if self.dout.ndim == 1:
            self.dout = self.dout.reshape(1, self.dout.shape[0])

        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, self.dout)
        self.db = np.sum(self.dout, axis = 0)

        return dx


#出力層
#恒等関数レイヤ
class liner:
    def __init__(self):
        self.y = None #出力

    def forward(self, x, t):
         self.y = LinerFunction(x)

         return self.y


#2乗和誤差レイヤ
class mean_squared_error:
    def __init__(self):
        self.loss = None
        self.t = None

    def function(self, x, t):
        self.x = x
        self.t = t
        self.y = MeanSquaredError(x, t)

        return self.y


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

