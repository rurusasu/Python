# cording :utf-8
import sys, os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.functions import _CallFunction, _CallClass
from common.gradient import numerical_gradient_2d
import numpy as np


class Sequential:
    def __init__(self):
        self.sequential = OrderedDict()
        self.func = {}
        self.history = {}
        #self.OutputBuff = []
        self.history['loss']     = []
        self.history['loss_ave'] = []
        #self.Output.history['val_loss'] = []
        #self.Output.history['acc']      = []
        #self.Output.history['val_acc']  = []
        self.i = 1


    def add(self, layer_name):
        #リストにレイヤの名前を代入
        self.sequential[self.i] = layer_name
        self.i += 1


    def compile(self, loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']):
        #self.LastLayer = globals()[loss]()
        self.func['loss'] = _CallClass('common.layers', loss)
        self.func['loss'] = self.func['loss']()
        self.func['optimizer'] = _CallClass('common.optimizer', optimizer)
        self.func['metrics'] = metrics


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
    def fit(self, input, test, batch_size, epochs, validation_data, epsilon=0.01, reg_lambda=0.01):
        plot = Plot(0, 1)

        IRS = input.shape[0]
        ICS = input.shape[1]
        
        #ValidationData
        #x_val_data = validation_data[0] # ValidationDataの行数を取得 返り値：整数
        #t_val_data = validation_data[1] # ValidationDataの列数を取得 返り値：整数
               
        #ValidationRow_size = x_val_data.shape[0]
        #ValidationCol_size = x_val_data.shape[1]
        
       

        #レイヤの行列を計算する
        y = ICS
        for layers in self.sequential.values():
            y = layers.fit(y, epsilon, reg_lambda)

        # メインルーチン
        for i in range(epochs):
            batch_mask = np.random.choice(IRS, batch_size, replace = False) #行数からbatch_sizeだけランダムに値を抽出 replace(重複)
            TrainingI_batch = input[batch_mask, 0:ICS] #全データからbatch_size分データを抽出
            TrainingT_batch = test[batch_mask]

            #x_val   = x_val_data[batch_mask, 0:ValidationCol_size]
            #t_val   = t_val_data[batch_mask]
            
            #print('#######    学習%d回目    ########' %Sequential.counter)
            #Sequential.counter += 1

            out_sum  = 0
            out_ave  = 0
            loss_sum = 0
            loss_ave = 0
            for j in range(batch_size):
                y = self.__predict__(TrainingI_batch[j, :]) # 学習を行う
                
                #####     誤差を保存する     #####
                loss = self.func['loss'].forward(y, TrainingT_batch[j])
                #loss = self.func['loss'].forward()
                loss_sum += loss
                self.history['loss'].append(loss)

            #out_ave  = out_sum  / batch_size
            loss_ave = loss_sum / batch_size
            plot.grah_plot(i+1, loss_ave)
            self.history['loss_ave'].append(loss_ave)

            
            ########################
            #####     追加     #####
            ########################
            #逆伝播を行うためにレイヤを反転
            revSequence = list(self.sequential.values())
            revSequence.reverse()

            #逆伝搬および重みの更新
            dout = self.func['loss'].backward(dout=1)
            for revLayer in revSequrnce:
                dout = revLayer.backward(dout)
            del revSequence
            self.sequential.reverse()
        
        print('loss = %f' %self.history['loss'][epochs-1])
        return self.history

    ########################################
    ########      内部関数       ###########
    ########################################
    def __predict__(self, data_set):
        for layers in self.sequential.values():
            data_set = layers.forward(data_set)

        return data_set


    def ValLoss(self, val_data, t_val):
        val_data = self.Predict(val_data)
        val_loss = self.LastLayer.forward(val_data, t_val)

        return val_loss


    def accuracy(self, x, t):
        y = self.Predict(x)
        #y = np.argmax(y, axis = 1)
        #if t.ndim != 1 : t = np.argmax(t, axis = 1)

        if y.shape != t.shape:
            y = y.reshape(-1, 1)
            t = t.reshape(-1, 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


class Plot:
    def __init__(self, x, y):
        #グラフの初期化
        self.fig, self.ax = plt.subplots(1, 1)
        self.x = []
        self.y = []

        self.x.append(x)
        self.y.append(y)

        self.lines,  = self.ax.plot(self.x, self.y)

    def grah_plot(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.lines.set_data(self.x, self.y)
        self.ax.set_xlim(0, len(self.x))
        self.ax.set_ylim(min(self.y), max(self.y))
        plt.pause(.01)


class Loss:
   #-------------------------------------------------
   # __init__:初期化を行う
   #     引数
   #     @self
   #     @loss_function :誤差を計算する関数を指定
   #-------------------------------------------------
    def __init__(self, loss_function):
        self.function = globals()[loss]()
        self.loss = 0
        self.loss_sum = []
    
   #-------------------------------------------------
   # loss:誤差を計算する
   #     引数
   #     @self
   #     @x             :NNからの出力
   #     @t             :テストデータ
   #     @sum           :誤差を記録するか選べる（0:記録させない, 1:記録させる）
   #-------------------------------------------------
    def loss(self, x, t, sum = 0):
        self.loss = self.function(x, t)
        if sum == 1:
            self.loss_sum.append(loss)
        return self.loss
    
   #-------------------------------------------------
   # ave_loss:誤差を計算する
   #     引数
   #     @self
   #     @batch_size    :バッチ数
   #-------------------------------------------------
    def ave_loss(self, batch_size):
        batch_loss = sum(self.loss_sum)
        return batch_loss / batch_size