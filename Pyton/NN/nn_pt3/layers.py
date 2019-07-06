#coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from nn_pt3.functions import *
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
    def __init__(self, W, b, epsilon, reg_lambda):
        # パラメータの設定
        self.W = W
        self.B = b
        self.dW = None # 重みの微分
        self.dB = None # バイアスの微分
        self.epsilon    = epsilon    # gradient descentの学習率
        self.reg_lambda = reg_lambda # regularizationの強さ
   
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

        self.W -= self.epsilon * self.dW
        self.B -= self.epsilon * self.dB

        dx = dx.reshape(self.original_x_shape) #逆伝播を入力信号の形に戻す
        return dx


#出力層
#2乗和誤差レイヤ
class MeanSquaredError:
    def __init__(self):
        self.loss = None #損失
        self.y    = None #linerの出力
        self.t    = None #教師データ

    def forward(self, x, t):
        #if t.shape != x.shape:
            #self.y = self.y.reshape(self.y.size, 1)
            #self.t = self.t.reshape(self.t.size, 1)
        self.loss = mean_squared_error(x, t)

        return self.loss

    
    def backward(self, loss, dout = 1):
        return loss


#####損失関数#####

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
   #-------------------------------------------------
   # __init__:初期化を行う
   #     引数
   #     @self
   #     @input_shape            :学習データの形状(タプル)
   #     変数
   #     @InputParams            :ユニット内での計算に必要なパラメータ(配列)             
   #     @InputParams['RowSize'] :入力データの行数
   #     @InputParams['ColSize'] :入力データの列数
   #-------------------------------------------------
    def __init__(self, input_shape):
        self.InputParams = {}
        self.InputParams['RowSize'] = None
        self.InputParams['ColSize'] = None
       
        if (len(input_shape)) == 1:   #もし、入力数が配列で指定されたとき
            self.InputParams['RowSize'] = input_shape[0]

        elif (len(input_shape) == 2):
            self.InputParams['RowSize'] = input_shape[0]
            self.InputParams['ColSize'] = input_shape[1]

        else:
            print("input_OverDimension")

   #-------------------------------------------------
   # unit:データに必要なユニット数を設定する
   #     引数
   #     @self
   #     @input_col_size         :入力するデータの列数
   #     @minibatch              :ミニバッチ数
   #     @counter                :何層目か示すために必要
   #     @epsilon                :学習率
   #-------------------------------------------------
    def unit(self, batch_or_minibatch_size, input_col_size, counter, epsilon, reg_lambda):
        #入力するデータの列数が、batch_or_minibatch_sizeと等しいか判定
        if self.InputParams['ColSize'] != input_col_size:
            if self.InputParams['RowSize'] == input_col_size: # RowSizeとColSizeを間違えて入力している可能性を判定
                self.InputParams['ColSize'] = self.InputParams['RowSize']

            elif self.InputParams['RowSize'] != input_col_size: # 等しくなければColSizeの値を更新
                self.InputParams['ColSize'] = input_col_size

        # 入力するデータの行数がbatch_or_minibatch_sizeと等しいか判定
        if self.InputParams['RowSize'] == batch_or_minibatch_size: # 等しければpass
            pass

        elif self.InputParams['RowSize'] != batch_or_minibatch_size: # 等しくなければRowSizeの値を更新
            self.InputParams['RowSize'] = batch_or_minibatch_size

        print('第%d層 - InputLayer' %counter)

        return self.InputParams['ColSize']


    def forward(self, input_data):
        out = np.reshape(input_data, [self.InputParams['RowSize'], self.InputParams['ColSize']])
        return out

    def backward(self, dout):
        pass


#全結合レイヤ
class Dense:
    def __init__(self,  Units, activation, weight_initializer='glorot_uniform', bias_initializer='zeros'):
        self.dense = OrderedDict()         #関数の辞書
        self.RevDense = None               #関数の辞書の反転(逆伝播で使用)
        self.activation = activation       #活性化関数名

        self.params = {}                 #ユニット内での計算に必要なパラメータの辞書
        self.params['Units']  = Units    #ユニットの数
        self.params['Weight'] = None     #重み
        self.params['Bias']   = None     #閾値
        self.params['epsilon'] = None    #学習率
        self.params['reg_lambda'] = None #regularizationの強さ

        #####   初期値の設定   #####
        self.initialisation   = InitParams()
        self.init_weight      = weight_initializer
        self.init_bias        = bias_initializer
        

    def initparams(self, BefLayer_Size, Unit_size):
        #K = 2
        #初期値の計算
        #weight =  K*(np.ones((input_size, self.params['Units']))*0.5 - np.random.rand(input_size, self.params['Units'])) #重み
        weight = InitParams.glorot_uniform(input_dim=BefLayer_Size, h_dim=self.params['Units'])
        bias   = np.zeros(self.params['Units'])                                                                           #閾値
                
        return weight, bias

    def unit(self, _, BefLayer_Size, counter, epsilon, reg_lambda):
        #初期値を設定
        self.params['epsilon'] = epsilon
        self.params['reg_lambda'] = reg_lambda
        self.params['Weight'] = self.initialisation.glorot_uniform(BefLayer_Size, self.params['Units'])
        self.params['Bias']   = np.zeros((1, self.params['Units']))  
        
        #レイヤの設定
        self.dense['Affine']     = globals()['affine'](self.params['Weight'], self.params['Bias'], epsilon, reg_lambda) #アフィン変換を行うレイヤをセット
        self.dense['Activation'] = globals()[self.activation]()                                    #活性化関数のレイヤをセット

        #レイヤの名前を表示
        print('第%d層 - AffineLayer' %counter)
        print('第%d層 - Activation %s' %(counter, self.activation))

        return self.params['Units']


    def forward(self, x):
        for layer in self.dense.values():
            x = layer.forward(x)

        return x


    def backward(self, dout):
        self.RevDense = list(self.dense.values()) #OrederedDictを使う場合、内部の値を入れ替える際はlistにする必要がある。
        self.RevDense.reverse()
        for RevLayer in self.RevDense:
            dout = RevLayer.backward(dout)

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


#誤差を計算
class Loss:
   #-------------------------------------------------
   # __init__:初期化を行う
   #     引数
   #     @self
   #     @loss_function :誤差を計算する関数を指定
   #     変数
   #     @Function      :誤差を計算する関数
   #     @ResultSum     :NN出力のbatch数分の合計
   #     @LossSum       :誤差のbatch数分の合計
   #-------------------------------------------------
    def __init__(self, loss_function):
        self.Function = globals()[loss_function]()
        self.LossShape = {}
        self.LossShape['Row'] = None
        self.LossShape['Col'] = None
        self.ResultSum = []
        self.LossSum   = []
  

   #-------------------------------------------------
   # forward:誤差を計算する
   #     引数
   #     @self
   #     @data          :NNからの出力
   #     @copm_data     :テストデータ(comper_dataの略　comper:比較)
   #     @sum           :誤差を記録するか選べる（0:記録させない, 1:記録させる）
   #     変数
   #     @ResultSum     :NN出力のbatch数分の合計
   #     @LossSum       :誤差のbatch数分の合計
   #-------------------------------------------------
    def forward(self, data, comp_data, minibatch, sum = 0):
        self.LossShape['Row'], self.LossShape['Col'] = data.shape
        loss = np.sum(self.Function.forward(data, comp_data)) / minibatch
        #if sum == 1:
            #self.ResultSum.append(data)
            #self.LossSum.append(loss)

        return loss
    
   #-------------------------------------------------
   # backward:逆伝播出力を計算する
   #     引数
   #     @self
   #     @batch_size    :バッチ数
   #     変数
   #     @BatchResult   :NN出力のbatch平均
   #     @BatchLoss     :誤差のbatch平均
   #     @BackSignal    :逆伝播信号
   #-------------------------------------------------
    def backward(self, loss_data, cycle):
        if(len(loss_data) == 1):
            BatchLoss  = sum(loss_data)
            BackSignal  = np.full((self.LossShape['Row'], self.LossShape['Col']), BatchLoss)
            BackSignal = self.Function.backward(BackSignal)
            

        elif(len(loss_data) != 1):
            BatchLoss  = sum(loss_data) / cycle
            BackSignal  = np.full((self.LossShape['Row'], self.LossShape['Col']), BatchLoss)
            BackSignal = self.Function.backward(BackSignal)

        return BatchLoss, BackSignal


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
 