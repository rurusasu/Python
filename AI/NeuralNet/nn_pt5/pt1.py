import sys, os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from common.sequential import Sequential
from common.layers import InputLayer, Dense
from common.functions import*


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