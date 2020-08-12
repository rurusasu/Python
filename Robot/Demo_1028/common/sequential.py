import sys, os
sys.path.append(os.getcwd())

import numpy as np
from common.functions import _CallClass, _CallFunction
from common.layers import *
from collections import OrderedDict


class Sequential:
    def __init__(self):
        #self.sequential = []
        self.sequential = OrderedDict()
        self.units = {}
        self.i = 1
        self.func = {}
        self.func['loss'] = None
        self.func['optimizer'] = None

        self.history = {}
        #self.history['loss'] = []
        self.history['loss_ave'] = []
        self.history['val_loss'] = []
        self.history['train_acc'] = []

    def add(self, layer_name):
        #リストにレイヤの名前を代入
        #self.sequential.append(layer_name)
        self.sequential[self.i] = layer_name
        self.units[self.i] = self.sequential[self.i].units
        self.i += 1

    def compile(self, loss, optimizer='sgd'):
        self.func['loss'] = _CallClass('common.layers', loss)
        self.func['loss'] = self.func['loss']()
        self.func['optimizer'] = _CallClass('common.optimizer', optimizer)
        self.func['optimizer'] = self.func['optimizer']()
        #self.LastLayer = globals()[loss]()

   #-------------------------------------------------
   # fit:学習のメイン
   #     引数
   #     @self
   #     @training_input:学習データ（入力）
   #     @training_test :教師データ
   #     @epochs        :エポック数
   #     @epsilon=0.01  :学習率（初期値0.01）
   #     変数
   #     @TrainingI       :TrainingInputの略
   #     @TrainigT        :TrainigTestの略
   #     @IRS             :InputRowSizeの略(training_inputの行数 返り値：整数)
   #     @ICS             :InputColSizeの略(training_inputの列数 返り値：整数)
   #     @batch_size      :バッチ数
   #     @TrainingI_batch :TrainingIからバッチ数個だけデータを抽出した行列
   #     @TrainingT_batch :TrainingTからバッチ数個だけデータを抽出した行列
   #-------------------------------------------------

    def fit(self, x, t, batch_size, epochs, validation=None):
        #ValidationData
        #if (validation != None):
            
        #x_val_data = validation_data[0] # ValidationDataの行数を取得 返り値：整数
        #t_val_data = validation_data[1] # ValidationDataの列数を取得 返り値：整数

        #ValidationRow_size = x_val_data.shape[0]
        #ValidationCol_size = x_val_data.shape[1]
        """
        x, t = self.__modeling__(x, t, self.units[1])
        if (validation != None and type(validation) == tuple):
            x_val, t_val = self.__modeling__(validation[0], validation[1])
        else:
            print('No validation')
        """
        #レイヤの行列を計算する
        y = np.zeros((batch_size, x.shape[1]))
        for layer in self.sequential.values():
            y = layer.fit(y)

        loop = int(x.shape[0] / batch_size)  # 繰り返し回数
        # メインルーチン
        for i in range(epochs):
            loss_sum = 0
            for j in range(loop):
                # 行数からbatch_sizeだけランダムに値を抽出 replace(重複)
                batch_mask = np.random.choice(
                    x.shape[0], batch_size, replace=False)
                x_batch = x[batch_mask]  # 全データからbatch_size分データを抽出
                t_batch = t[batch_mask]
                """
                if (validation != None):
                    x_val   = x_val[batch_mask]
                    t_val   = t_val[batch_mask]
                """
                loss = self.gradient(x_batch, t_batch) # 誤差の計算
                loss_sum += (loss / batch_size)
            loss_ave = loss_sum / loop                 # 誤差の平均値の計算
            self.history['loss_ave'].append(loss_ave)  # 誤差の保存
            print('学習%d回目  --loss:%f' % (i+1, loss_ave))
            
            #---------------------------
            # validation誤差の計算
            #---------------------------
            if(validation != None):
                val_loss = self.loss(validation[0], validation[1])
                val_loss_ave = val_loss / validation[0].shape[0]
                self.history['val_loss'].append(val_loss_ave)

            #---------------------------
            # 正解率の計算
            #---------------------------
            

        print('loss=%f, val=%f' % (self.history['loss_ave'][epochs-1], self.history['val_loss'][epochs-1]))
        return self.history


    ########################################
    ########      内部関数       ###########
    ########################################
    def __modeling__(self, x, t, size):
        # 入力層の行数と入力データの行数が等しいとき
        # もしくは入力層の行数と入力データの列数が等しいとき
        if (x.shape[0] == size):
            x = x.T
            t = t.T
            return x, t
        elif(x.shape[1] == size):
            return x, t
        # どちらとも等しくないとき
        else:
            print('InpuLayer fit \
                    Data Input Error')
            return None, None


    def predict(self, x):
        for layer in self.sequential.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.func['loss'].forward(y, t)

    def gradient(self, x, t):
        #forward
        loss = self.loss(x, t)

        # backward
        #逆伝播を行うためにレイヤを反転
        layers = list(self.sequential.values())
        layers.reverse()

        #逆伝搬および重みの更新
        dout = 1
        dout = self.func['loss'].backward()
        for layer in layers:
            dout = layer.backward(dout)
        del layers

        return loss


    def accuracy(self, x, t):
        y = self.predict(x)
        
        # 相対誤差
        acc = np.sum(abs(y-t)) / float(x.shape[0])
        #acc = np.sum(y == t) / float(x.shape[0])
        return acc
