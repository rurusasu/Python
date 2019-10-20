import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from nn_pt1.functions import *
from nn_pt1.layers import *
from collections import OrderedDict

class output: pass


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


class Sequential:
    counter  = 1

    def __init__(self):
        self.sequential = []
        self.Output = output()
        self.Output.history = {}

        self.OutputBuff = []
        self.Output.history['loss']     = []
        self.Output.history['loss_ave'] = []
        #self.Output.history['val_loss'] = []
        #self.Output.history['acc']      = []
        #self.Output.history['val_acc']  = []

    def add(self, layer_name):
        #リストにレイヤの名前を代入
        self.sequential.append(layer_name)

    def compile(self, loss):
        self.LastLayer = globals()[loss]()


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
    def fit(self, training_input, training_test, batch_size, epochs, validation_data, epsilon=0.01, reg_lambda=0.01):
        plot = Plot(0, 1)

        IRS = training_input.shape[0]
        ICS = training_input.shape[1]
        
        #ValidationData
        #x_val_data = validation_data[0] # ValidationDataの行数を取得 返り値：整数
        #t_val_data = validation_data[1] # ValidationDataの列数を取得 返り値：整数
               
        #ValidationRow_size = x_val_data.shape[0]
        #ValidationCol_size = x_val_data.shape[1]
        
       

        #レイヤの行列を計算する
        y = ICS
        for layers in self.sequential:
            y = layers.unit(y, Sequential.counter, epsilon, reg_lambda)
            Sequential.counter += 1

        Sequential.counter = 1
        # メインルーチン
        for i in range(epochs):
            batch_mask = np.random.choice(IRS, batch_size, replace = False) #行数からbatch_sizeだけランダムに値を抽出 replace(重複)
            TrainingI_batch = training_input[batch_mask, 0:ICS] #全データからbatch_size分データを抽出
            TrainingT_batch = training_test[batch_mask]

            #x_val   = x_val_data[batch_mask, 0:ValidationCol_size]
            #t_val   = t_val_data[batch_mask]
            
            print('#######    学習%d回目    ########' %Sequential.counter)
            Sequential.counter += 1

            out_sum  = 0
            out_ave  = 0
            loss_sum = 0
            loss_ave = 0
            for j in range(batch_size):
                #output = self.Predict(TrainingI_batch[j, :]) # 学習を行う
                output = self.Predict(TrainingI_batch[j])
                
                out_sum += output
                #####     誤差を保存する     #####
                loss = 0
                loss = self.LastLayer.forward(output, TrainingT_batch[j])
                loss_sum += loss
                self.Output.history['loss'].append(loss)

            loss_ave = loss_sum / batch_size
            out_ave  = out_sum  / batch_size
            plot.grah_plot(i+1, loss_ave)
            self.Output.history['loss_ave'].append(loss_ave)

            
            ########################
            #####     追加     #####
            ########################
            #逆伝播を行うためにレイヤを反転
            self.sequential.reverse()

            #逆伝搬および重みの更新
            dout = self.LastLayer.backward(out_ave, loss_ave)
            for layer in self.sequential:
                dout = layer.backward(dout)
            
            self.sequential.reverse()
        
        print('loss = %f' %self.Output.history['loss'][epochs-1])
        return self.Output


    ########################################
    ########      内部関数       ###########
    ########################################
    def Predict(self, data_set):
        for layers in self.sequential:
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



#訓練データの読み込み
data = np.loadtxt(
    "save_data.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )

#テストデータの読み込み
test = np.loadtxt(
    "test_data.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )



#読み込んだデータを学習用にコピーする
data_1 = data_copy(data)
test_1 = data_copy(test)

#標準化
data_ = data_std(data_1)
test_ = data_std(test_1)

#正規化
data_ = data_nom(data_)
test_ = data_nom(test_)

#訓練データのセット
training_input = data_[:, 0:2] #学習データをセット
training_test = data_[:, 2]   #教師データをセット

#テストデータのセット
x_test  = test_[:, 0:2] #入力データをセット
t_test  = test_[:, 2]   #正解データをセット


module = Sequential()
module.add(InputLayer(input_shape = 2))
#module.add(Dense(50, activation = 'sigmoid'))
#module.add(Dense(50, activation = 'sigmoid'))
module.add(Dense(50, activation = 'relu'))
module.add(Dense(50, activation = 'relu'))
module.add(Dense(1,  activation = 'liner'))
module.compile(loss = 'mean_squared_error')

#学習
epochs = 20
batch_size = 128

# Gradient descent parameters (数値は一般的に使われる値を採用) 
epsilon = 0.01    # gradient descentの学習率
reg_lambda = 0.01 # regularizationの強さ 

history = module.fit(training_input, training_test, batch_size=batch_size, epochs=epochs, validation_data = (x_test, t_test), epsilon=epsilon, reg_lambda=reg_lambda)

#lossグラフ
loss     = history.history['loss_ave']
#val_loss = history.history['val_loss']

nb_epoch = len(loss)
plt.plot(range(nb_epoch), loss,     marker = '.', label = 'loss')
#plt.plot(range(nb_epoch), val_loss, marker = '.', label = 'val_loss')
plt.legend(loc = 'best', fontsize = 10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
