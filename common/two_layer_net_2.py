# coding: utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers_2 import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        
        #重みとバイアス(閾値)の初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()

        self.lastLayer = IdentityLayer()


    #推論を行う
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    #x:入力データ, t:教師データ
    #損失関数の値を求める
    def loss(self, x, t):
        y = self.predict(x)  #ニューロン内の各レイヤで計算を行う
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    #重みパラメータに対する勾配を数値微分によって求める
    #x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


    def forward(self, x, t):
        #forward
        Loss = self.loss(x, t) #順伝播の損失関数

        return Loss


    #重みパラメータに対する勾配を誤差逆伝搬法によって求める
    def gradient(self, x, t, Err_Total):
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout, Err_Total)

        layers = list(self.lastLayer.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #設定
        grads = {}
        grads['W1'] = self.layers['Affien1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
