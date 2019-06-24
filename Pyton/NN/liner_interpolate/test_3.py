import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.functions import *
from common.layers_2p import *
from collections import OrderedDict

class output: pass

class Sequential:
    counter  = 1
    counter2 = 1

    def __init__(self):
        self.sequential = []
        self.Output = output()
        self.Output.history = {}

        self.Output.history['loss']     = []
        self.Output.history['val_loss'] = []
        self.Output.history['acc']      = []
        self.Output.history['val_acc']  = []

    def add(self, layer_name):
        #リストにレイヤの名前を代入
        self.sequential.append(layer_name)

    def compile(self, loss):
        self.LastLayer = globals()[loss]()


    def fit(self, learning_data, training_data, batch_size, epochs, validation_data, epsilon=0.01, reg_lambda=0.01):
        #TrainingData
        x = learning_data
        t = training_data
        TrainRow_size = x.shape[0] # learning_dataの行数を取得 返り値：整数
        TrainCol_size = x.shape[1] # learning_dataの列数を取得 返り値：整数
        
        #ValidationData
        x_val_data = validation_data[0] # ValidationDataの行数を取得 返り値：整数
        t_val_data = validation_data[1] # ValidationDataの列数を取得 返り値：整数
               
        ValidationRow_size = x_val_data.shape[0]
        ValidationCol_size = x_val_data.shape[1]
        
        cycle = int(TrainRow_size/batch_size)

        #レイヤの行列を計算する
        y = TrainCol_size
        for layers in self.sequential:
            y = layers.unit(y, Sequential.counter, epsilon, reg_lambda)
            Sequential.counter += 1

        # メインルーチン
        for i in range(epochs):
            batch_mask = np.random.choice(TrainRow_size, batch_size, replace = False) #行数からbatch_sizeだけランダムに値を抽出 replace(重複)
            x_batch = x[batch_mask, 0:TrainCol_size] #全データからbatch_size分データを抽出
            t_batch = t[batch_mask]

            x_val   = x_val_data[batch_mask, 0:ValidationCol_size]
            t_val   = t_val_data[batch_mask]
                                       
            #####     推論を行う     #####
            print('#######    学習%d回目    ########' %Sequential.counter2)
            output = self.Predict(x_batch)
            Sequential.counter2 = Sequential.counter2 + 1
                
            #####     誤差を保存する     #####
            loss = 0
            loss = self.LastLayer.forward(output, t_batch)
            self.Output.history['loss'].append(loss)

            ########################
            #####     追加     #####
            ########################
            #逆伝播を行うためにレイヤを反転
            self.sequential.reverse()

            #逆伝搬および重みの更新
            dout = 1
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
learning_data = data_[:, 0:2] #学習データをセット
training_data = data_[:, 2]   #教師データをセット

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

history = module.fit(learning_data, training_data, batch_size=batch_size, epochs=epochs, validation_data = (x_test, t_test), epsilon=epsilon, reg_lambda=reg_lambda)

#lossグラフ
loss     = history.history['loss']
#val_loss = history.history['val_loss']

nb_epoch = len(loss)
plt.plot(range(nb_epoch), loss,     marker = '.', label = 'loss')
#plt.plot(range(nb_epoch), val_loss, marker = '.', label = 'val_loss')
plt.legend(loc = 'best', fontsize = 10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()