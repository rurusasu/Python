#coding: utf-8
import sys, os
sys.path.append(os.getcwd())
import numpy as np
from common.functions import _CallFunction, _CallClass
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
    
    def forward(self, x):
        self.mask = (x <= 0) #xが0以下ならFalseを返す(0より大きければTrue)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        delta = dout

        return delta


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

        return delta


#恒等関数レイヤ
class liner:
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
    def __init__(self, W, b, lr):
        # パラメータの設定
        self.W = W
        self.B = b
        self.dW = None # 重みの微分
        self.dB = None # バイアスの微分
        self.lr = lr   # gradient descentの学習率
        self.x = None
        self.original_x_shape = None


    def forward(self, x):
        #テンソル対応
        self.original_x_shape = x.shape #元の形を記憶させる
        x = x.reshape(x.shape[0], -1)   #奥行き方向の幅を固定しつつ、行列の大きさを変更
        self.x = x
        
        out = self.x.dot(self.W) + self.B

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.dB = np.sum(dout, axis = 0)

        self.W -= self.lr * self.dW
        self.B -= self.lr * self.dB

        dx = dx.reshape(self.original_x_shape) #逆伝播を入力信号の形に戻す
        return dx


#####損失関数#####
#2乗和誤差レイヤ
class mean_squared_error:
    def __init__(self):
        #self.loss = None #損失
        self.y = None  # linerの出力
        self.t = None  # 教師データ
        self.func = _CallFunction('common.functions', 'mean_squared_error')

    def forward(self, y, t):
        self.y = y
        self.t = t
        loss = self.func(self.y, self.t)

        return loss

    def backward(self, dout=1):
        if dout == 1:
            batch_size = self.t.shape[0]
            dout = (self.y - self.t) / batch_size
        return dout


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
class Input:
    def __init__(self, input_shape):
        #self.units  = input_shape #ユニットの数
        if (type(input_shape) == int):
            self.units = input_shape
        elif (type(input_shape) == tuple):
            self.units = input_shape[0]
        else:
            print('error InputLayer!')

    def fit(self, x):
        return x

    def forward(self, x):
        return x

    def backward(self, dout):
        pass


#全結合レイヤ
class Dense:
    def __init__(self,  units, activation, weight_initializer='he', bias_initializer='zeros'):
        if (units != None and type(units) == int):
            self.units = units
        # 損失関数名
        self.activation = activation  # 活性化関数名
        # 重みとバイアスの初期化関数名
        self.initializer = {}
        self.initializer['W'] = weight_initializer
        self.initializer['b'] = bias_initializer
        # 重みとバイアスの初期値
        self.params = {}
        self.params['W'] = None  # 重み
        self.params['b'] = None  # 閾値
        # 内部レイヤ
        self.func = OrderedDict()  # 関数の辞書
        self.func['Affine'] = None
        self.func['Activation'] = None
        self.RevDense = None               #関数の辞書の反転(逆伝播で使用)
  

    def fit(self, x):
        #初期値を設定
        self.params['W'] = self.__InitWeight__(
            x.shape[1], self.units, 50, self.initializer['W'])
        self.params['b'] = self.__InitBias__(self.units, self.initializer['b'])

        #レイヤの設定
        self.__SetFunc__(
            self.params['W'], self.params['b'], self.activation, lr=0.01)

        x = self.forward(x)

        return x

    def __InitWeight__(self, row, col, n, weight_initializer='he'):
        """
        重みの初期設定

        Parameters
        ----------
        row : 重みの行数
        col : 重みの列数
        n   : 前の層のユニット数
        weight_initializer : 重みの標準偏差を指定
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        """
        if str(weight_initializer).lower() in ('relu', 'he'):
            weight_initializer = 'he_nomal'
        elif str(weight_initializer).lower() in ('sigmoid', 'xavier'):
            weight_initializer = 'glorot_uniform'
        method = _CallFunction('common.weight', weight_initializer)
        scale = method(n)
        return scale * np.random.randn(row, col)

    def __InitBias__(self, row, bias_initializer='zeros'):
        """
        閾値の初期設定

        Parameters
        ----------
        row : 重みの行数
        bias_initializer : biasを指定
        """
        method = _CallFunction('common.bias', bias_initializer)
        return method(row)


    def __SetFunc__(self, weight, bias, activation, lr):
        """
        ユニット内部関数をセットする

        Parameters
        ----------
        lr : float
            学習率
        """
        #レイヤの設定
        self.func['Affine'] = globals()['affine'](
            self.params['W'], self.params['b'], lr)  # アフィン変換を行うレイヤをセット
        self.func['Activation'] = globals()[self.activation]()


    def forward(self, x):
        for layer in self.func.values():
            x = layer.forward(x)

        return x


    def backward(self, dout):
        revDense = list(self.func.values()) #OrederedDictを使う場合、内部の値を入れ替える際はlistにする必要がある。
        revDense.reverse()
        for revLayer in revDense:
            dout = revLayer.backward(dout)
        del revDense

        return dout


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout, counter):
        print("第%d層 BatchNome" %counter)
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
 
