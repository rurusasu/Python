#coding: utf-8

import sys, os
sys.path.append(os.getcwd())
from collections import OrderedDict
from common.functions import _CallFunction, _CallClass
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


#2乗和誤差レイヤ
class mean_squared_error:
    def __init__(self):
        self.loss = None  # 損失
        self.y = None  # linerの出力
        self.t = None  # 教師データ
        self.func = _CallFunction('common.functions', 'mean_squared_error')

    def forward(self, y, t):
        #if t.shape != y.shape:
            #self.y = y.reshape(self.y.size, 1)
            #self.t = t.reshape(self.t.size, 1)
        self.y = y
        self.t = t
        self.loss = self.func(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        if dout == 1:
            if self.y.size == self.t.size:
                batch_size = self.t.shape[0]
                if batch_size == 1:
                    dout = self.y - self.t
                else:
                    dout = (self.y - self.t) / batch_size

        return dout


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


    def fit(self, rear_node):
        self.units = (self.units, rear_node)


    def forward(self, data):
        """
        入力層のノード数と入力データの行数を比較する

        Parameter
        ---------
        data : numpy.ndarray
            入力データ

        Return
        ------
        data : numpy.ndarray
            出力データ
        """
        data_rowSize = data.shape[0]
        # 入力データの行数と入力層のノード数が
        # 同じ場合
        if (data_rowSize == self.units[0]):
            return data
        # 入力データの行数が多い場合
        elif (data_rowSize > self.units[0]):
            mask = np.random.choice(data_rowSize, self.units[0])
            data = data[mask, 0:]
            return data
        # ノード数が多い場合
        else:
            print('batch_size < InputLayer')
            return None

        #batch_mask = np.random.choice(self.units[0], self.units[1], replace=False)
    

    def backward(self, dout):
        pass


    def _GetParams(self):
        print('----------------------')
        print(self.units)
        print('-----------------------')
        


class Dense:
    def __init__(self, Units, activation='relu', weight_initializer='he', bias_initializer='zeros'):
        if (Units != None and type(Units) == int):
            self.units = Units
        # 損失関数名
        self.activation = activation
        # 重みとバイアスの初期化関数名
        self.initializer  = {} 
        self.initializer['W'] = weight_initializer
        self.initializer['b'] = bias_initializer
        # 重みとバイアスの初期値
        self.params = {} 
        self.params['W'] = None
        self.params['b'] = None
        # 内部レイヤ
        self.function = OrderedDict() # 関数の辞書
        self.function['Affine'] = None
        self.function['Activation'] = None
        # 更新用の重みとバイアス
        self.diffParams = {}
        self.diffParams['W'] = None
        self.diffParams['b'] = None


    def fit(self, rear_node):
        """
        compile関数から呼び出される処理
        """
        # 重みの初期化
        self.__InitWeight__(rear_node, self.initializer['W'])
        self.__InitBias__(rear_node, self.initializer['b'])
        
        # 内部レイヤを構築
        self.__SetFunc__(lr=0.01)

    def __InitWeight__(self, rear_node, weight_initializer='he'):
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
        method = _CallFunction('common.weight', weight_initializer)
        scale = method(self.units)
        self.params['W'] = scale * np.random.randn(self.units, rear_node)

    def __InitBias__(self, rear_node, bias_initializer='zeros'):
        """
        閾値の初期設定

        Parameters
        ----------
        bias_initializer : biasを指定
        rear_node        : 次のnodeの行数
        """
        method = _CallFunction('common.bias', bias_initializer)
        self.params['b'] = method(rear_node)

    def __SetFunc__(self, lr):
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
    

    def forward(self, x):
        for layer in self.function.values():
            x = layer.forward(x)
        return x


    def backward(self, dout):
        revDense = list(self.function.values())
        revDense.reverse()
        for revLayers in revDense:
            dout = revLayers.backward(dout)
        del revDense

        self._optimizer()

        return dout


    def _optimizer(self, optimizer='sgd', loss=1, lr=0.01):
        self.diffParams['W'] = self.function['Affine'].dW
        self.diffParams['b'] = self.function['Affine'].dB
        # クラスのインスタンスを作成
        class_def = _CallClass('optimizer', optimizer)
        obj = class_def()

        obj.update(self.params, self.diffParams)


    def _GetParams(self):
        print('-----------------------------------')
        print('activation  = ' + self.activation)
        print('WeightInit  = ' + self.initializer['W'])
        print('WeightShape = ')
        print(self.params['W'].shape)
        print('BiasInit    = ' + self.initializer['b'])
        print('BiasShape   = ')
        print(self.params['b'].shape)
        print('function_1  = ')
        print(self.function['Affine'])
        print('function_2  = ')
        print(self.function['Activation'])
        print('dW = ')
        print(self.diffParams['dW'])
        print('db = ')
        print(self.diffParams['db'])
        print('-----------------------------------')


if __name__ == "__main__":
    #print(InputLayer(2))
    #print(InputLayer((2, 3)))
    
    dense = Dense(50)
    #Dense(weight_initializer='relu')
    dense.compile(50)
    dense._optimizer()
