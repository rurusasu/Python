import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 重み(ガウス分布で初期化)

    def predict(self, x):
        return np.dot(x, self.W) #重みと入力との積

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z) # 損失関数(ソフトマックス)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W) # 重みパラメータ

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print(np.argmax(p)) # 最大値のインデックス

t = np.array([0, 0, 1]) # 正解ラベル
print(net.loss(x, t))