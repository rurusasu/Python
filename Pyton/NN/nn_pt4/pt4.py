import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from nn_pt4.functions import *
from nn_pt4.layers import *
from collections import OrderedDict

class output: pass


class Plot:
    def __init__(self, x, y):
        #グラフの初期化
        self.fig, (self.axL, self.axR) = plt.subplots(ncols=2, figsize=(10, 4))
        self.x = []
        self.y = []

        self.x.append(x)
        self.y.append(y)

        self.lines_L,  = self.axL.plot(self.x, self.y)
        self.lines_R,  = self.axR.plot(self.x, self.y)

    def grah_plot_L(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.lines_L.set_data(self.x, self.y)
        self.axL.set_xlim(0, len(self.x))
        self.axL.set_ylim(min(self.y), max(self.y))
        plt.pause(.01)

    def grah_plot_R(self, x):
        loss = x
        nb_epoch = len(loss)
        #self.lines_R.set_data(range(nb_epoch), loss,  marker = '.', label = 'loss')
        self.lines_R.set_data(range(nb_epoch), loss)
        self.axR.set_xlim(0, nb_epoch)
        self.axR.set_ylim(min(loss), max(loss))
        self.axR.set_xlabel('epoch')
        self.axR.set_ylabel('loss')
        self.axR.grid(False)
        plt.pause(.01)


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
        self.Output.history['train_acc']      = []
        #self.Output.history['val_acc']  = []

    def add(self, layer_name):
        #リストにレイヤの名前を代入
        self.sequential.append(layer_name)

    def compile(self, loss):
        #self.LastLayer = globals()[loss]()
        self.loss = Loss(loss)

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
        plot = Plot(0, 0.125)

        IRS = training_input.shape[0]
        ICS = training_input.shape[1]
       
        minibatch = 1
        #ValidationData
        #x_val_data = validation_data[0] # ValidationDataの行数を取得 返り値：整数
        #t_val_data = validation_data[1] # ValidationDataの列数を取得 返り値：整数
               
        #ValidationRow_size = x_val_data.shape[0]
        #ValidationCol_size = x_val_data.shape[1]
        
       

        #レイヤの行列を計算する
        y = ICS
        for layers in self.sequential:
            y = layers.unit(minibatch, y, Sequential.counter, epsilon, reg_lambda)
            Sequential.counter += 1

        iter = int(IRS /batch_size) + 1 
        Sequential.counter = 1
        # メインルーチン
        for i in range(epochs):
            print('#######    学習%d回目    ########' %Sequential.counter)
            Sequential.counter += 1

            for j in range(iter):
                batch_mask = np.random.choice(IRS, batch_size, replace = False) #行数からbatch_sizeだけランダムに値を抽出 replace(重複)
                TrainingI_batch = training_input[batch_mask, 0:ICS] #全データからbatch_size分データを抽出
                TrainingT_batch = training_test[batch_mask]

                #x_val   = x_val_data[batch_mask, 0:ValidationCol_size]
                #t_val   = t_val_data[batch_mask]
            


                for k in range(batch_size):
                    output = self.Predict(TrainingI_batch[k, :]) # 学習を行う

                    #####     誤差を計算する     #####
                    loss = sum(mean_squared_error(output, TrainingT_batch[k]))
                    #self.Output.history['loss'].append(sum(loss))
                    self.Output.history['loss'].append(sum(loss)/float(output.shape[0]))

            BatchLoss = sum(self.Output.history['loss']) / (IRS)
            plot.grah_plot_L(i+1, BatchLoss)
            self.Output.history['loss_ave'].append(BatchLoss)
            self.Output.history['loss'] = []
            
            #逆伝播を行うためにレイヤを反転
            self.sequential.reverse()

            #逆伝搬および重みの更新
            BackSignal = sum(self.Output.history['loss']) / batch_size
            for layer in self.sequential:
                BackSignal = layer.backward(BackSignal)
            
            self.sequential.reverse()
        
            # 正解率の計算
            TrainAcc = self.accuracy(TrainingI_batch[0, :], TrainingT_batch)
            self.Output.history['train_acc'].append(TrainAcc)
            

        print('loss = %f' %self.Output.history['loss_ave'][epochs-1])
        print('train_acc = %f' %self.Output.history['train_acc'][epochs-1])

        plot.grah_plot_R(self.Output.history['loss_ave'])
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

        #accuracy = np.sum(y == t) / float(x.shape[0])
        accuracy = np.sum(y - t) / float(x.shape[0])

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
module.add(InputLayer(input_shape = (2, )))
#module.add(Dense(50, activation = 'sigmoid'))
#module.add(Dense(50, activation = 'sigmoid'))
module.add(Dense(50, activation = 'relu'))
module.add(Dense(50, activation = 'relu'))
module.add(Dense(1,  activation = 'liner'))
module.compile(loss = 'MeanSquaredError')

#学習
epochs = 5
batch_size = 32

# Gradient descent parameters (数値は一般的に使われる値を採用) 
epsilon = 0.01    # gradient descentの学習率
reg_lambda = 0.01 # regularizationの強さ 

history = module.fit(training_input, training_test, batch_size=batch_size, epochs=epochs, validation_data = (x_test, t_test), epsilon=epsilon, reg_lambda=reg_lambda)


loss = history.history['loss_ave']
'''
nb_epoch = len(loss)
plt.plot(range(nb_epoch), loss,  marker = '.', label = 'loss')
#plt.legend(loc = 'best', fontsize = 10)
plt.grid(False)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
'''
