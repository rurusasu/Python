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
    def __init__(self, W, b):
        # パラメータの設定
        self.W = W
        self.B = b
        self.dW = None # 重みの微分
        self.dB = None # バイアスの微分
        #self.epsilon    = epsilon    # gradient descentの学習率
        #self.reg_lambda = reg_lambda # regularizationの強さ
   
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

        self.W -= 0.01 * self.dW
        self.B -= 0.01 * self.dB

        dx = dx.reshape(self.original_x_shape) #逆伝播を入力信号の形に戻す
        return dx


#出力層
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
        if type(self.t) == float:
            dx = (self.y - self.t)

        elif self.t.size == self.y.size:
            batch_size = self.t.shape[0]
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arrange(batch_size), self.t] -= 1
            dx = dx /batch_size

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
        self.params = {}                    #ユニット内での計算に必要なパラメータの辞書
        self.params['units']  = input_shape #ユニットの数
        self.Input_Col_Size = None
        self.input          = None
       
        #if len(input_shape) == 1:   #もし、入力数が配列で指定されたとき
            #self.input = 1
            #pass
        #elif len(input_shape) == 2:
            #self.input = input_shape[1]


    def fit(self, Col_Size, epsilon, reg_lambda):
        return self.params['units']

    def forward(self, input_data):
        out = np.reshape(input_data, [-1, self.params['units']])
        return out
        #return input_data

    def backward(self, dout):
        pass


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
    def __init__(self,  units, activation='relu', weight_initializer='glorot_uniform', bias_initializer='zeros'):
        if (units != None and type(units) == int):
            self.units = units
        #活性化関数名
        self.activation = activation
        # 重みとバイアスの初期化関数名
        self.initializer = {}
        self.initializer['W'] = weight_initializer
        self.initializer['b'] = bias_initializer
        # 重みとバイアスの初期値
        self.params = {}
        self.params['W'] = None
        self.params['b'] = None
        # 内部計算
        self.func = OrderedDict()  #関数の辞書
        self.func['Affine'] = None
        self.func['Activation'] = None

        #self.RevDense = None               #関数の辞書の反転(逆伝播で使用)
        

        self.params['epsilon'] = None    #学習率
        self.params['reg_lambda'] = None #regularizationの強さ

        #####   初期値の設定   #####
        self.initialisation   = InitParams()
        self.init_weight      = weight_initializer
        self.init_bias        = bias_initializer
        

    def __InitWeight__(self, BefNode, weight_initializer='he'):
        """
        重みの初期設定

        Parameters
        ----------
        BefNode : 前の層のNode数
        weight_initializer : 重みの標準偏差を指定
            'relu'または'he'を指定した場合は「heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」
        """
        if str(weight_initializer).lower() in ('relu', 'he'):
            self.initializer['W'] = 'he_nomal'
        elif str(weight_initializer).lower() in ('sigmoid', 'xavier'):
            self.initializer['W'] = 'glorot_uniform'
        method = _CallFunction('common.weight', weight_initializer)
        scale = method(self.units)
        self.params['W'] = scale * np.random.randn(BefNode, self.units)
        #weight = InitParams.glorot_uniform(input_dim=BefLayer_Size, h_dim=self.params['Units'])
        #bias   = np.zeros(self.params['Units'])  #閾値
        #return weight, bias


    def __InitBias__(self, units, bias_initializer='zeros'):
        """
        閾値の初期設定

        Parameters
        ----------
        units : この層のunit数
        bias_initializer : biasを指定
        """
        method = _CallFunction('common.bias', bias_initializer)
        self.params['b'] = method(units)


    def __SetFunc__(self, activation):
        """
        ユニット内部関数をセットする

        Parameters
        ----------
        lr : float
            学習率
        """
        self.func['Affine']     = globals()['affine'](self.params['W'], self.params['b']) #アフィン変換を行うレイヤをセット
        self.func['Activation'] = globals()[activation]()


    def fit(self, BefLayer_Size, epsilon, reg_lambda):
        #初期値を設定
        #self.params['epsilon'] = epsilon
        #self.params['reg_lambda'] = reg_lambda
        self.__InitWeight__(BefLayer_Size, self.initializer['W'])
        self.__InitBias__(self.units, self.initializer['b'])
        self.__SetFunc__(self.activation)
        #self.params['Weight'] = self.initialisation.glorot_uniform(BefLayer_Size, self.params['Units'])
        #self.params['Bias']   = np.zeros((1, self.params['Units']))  
        
        #レイヤの設定
        #self.dense['Affine']     = globals()['affine'](self.params['Weight'], self.params['Bias'], epsilon, reg_lambda) #アフィン変換を行うレイヤをセット
        #self.dense['Activation'] = globals()[self.activation]()                                    #活性化関数のレイヤをセット

        #レイヤの名前を表示
        #print('第%d層 - AffineLayer' %counter)
        #print('第%d層 - Activation %s' %(counter, self.activation))

        return self.units


    def forward(self, x):
        for layer in self.func.values():
            x = layer.forward(x)

        return x


    def backward(self, dout):
        revDense = list(self.func.values()) #OrederedDictを使う場合、内部の値を入れ替える際はlistにする必要がある。
        revDense.reverse()
        for revLayer in self.revDense:
            dout = revLayer.backward(dout)
        del revDense

        return dout


#重みの初期値計算
class InitParams:
    def __init__(self):
        pass

    #重みの初期値
    #Xavierの一様分布
    def glorot_uniform(self, input_dim, h_dim):
        np.random.seed(0)
        weight = np.random.randn(input_dim, h_dim) / np.sqrt(input_dim)

        return weight

    #Heの初期値
    def he_nomal(self, input_dim, h_dim):
        weight = np.sqrt(2) * np.random.randn(input_dim, h_dim) / np.sqrt(input_dim)

        return weight


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
 