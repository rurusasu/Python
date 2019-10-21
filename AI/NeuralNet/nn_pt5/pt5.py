import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.sequential import Sequential
from common.layers import*
from common.functions import*


#訓練データの読み込み
data = np.loadtxt(
    "./data/learn.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )

#テストデータの読み込み
test = np.loadtxt(
    "./data/test.csv", #読み込むファイル名(例"save_data.csv")
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
input = data_ #学習データをセット
test = test_  #教師データをセット

#テストデータのセット
#x_test  = test_ #入力データをセット
#t_test  = test_ #正解データをセット


module = Sequential()
module.add(InputLayer(input_shape = 3))
module.add(Dense(50, activation = 'relu'))
module.add(Dense(50, activation = 'relu'))
module.add(Dense(3,  activation = 'liner'))
module.compile(loss = 'mean_squared_error')

#学習
epochs = 100
batch_size = 128

# Gradient descent parameters (数値は一般的に使われる値を採用) 
epsilon = 0.01    # gradient descentの学習率
reg_lambda = 0.01 # regularizationの強さ 

history = module.fit(input, test, batch_size=batch_size, epochs=epochs)

#lossグラフ
loss     = history['loss_ave']
#val_loss = history.history['val_loss']

nb_epoch = len(loss)
plt.plot(range(nb_epoch), loss,     marker = '.', label = 'loss')
#plt.plot(range(nb_epoch), val_loss, marker = '.', label = 'val_loss')
plt.legend(loc = 'best', fontsize = 10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()