# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import RMSprop

import numpy as np
import matplotlib.pyplot as plt


#データコピー
def data_copy(data):
    dim = data.ndim  # 受け取ったdataの次元を確認

    if dim == 1:  # dataが1次元(配列)のとき
        return data
    else:  # dataが2次元(行列)のとき
        data_copy = np.empty_like(data)  # dataと同じ大きさの空の行列を作成
        col = data.shape[1]  # dataの列数を取得

        for i in range(col):
            data_copy[:, i] = data[:, i]  # 対応する列に値をコピー

        return data_copy


#標準化
def data_std(data):
    dim = data.ndim  # 受け取ったdataの次元を確認

    if dim == 1:  # dataが1次元(配列)のとき
        return (data - data.mean()) / data.std()
    else:  # dataが2次元(行列)のとき
        data_std = np.empty_like(data)  # dataと同じ大きさの空の行列を作成
        row = data.shape[0]  # dataの行数を取得
        col = data.shape[1]  # dataの列数を取得

        #for i in range(row):
        #for j in range(col):
        #data_std[i, j] =(data[i, j] - data[i, j].mean()) / data[i, j].std() #対応する列を標準化

        for i in range(col):
            data_std[:, i] = (data[:, i] - data[:, i].mean()
                              ) / data[:, i].std()

        return data_std


#正規化
def data_nom(data):
    dim = data.ndim  # 受け取ったdataの次元を確認

    if dim == 1:  # dataが1次元(配列)のとき
        return data_1[:, 0] / max(abs(data_1[:, 0]))
    else:  # dataが2次元(行列)のとき
        data_nom = np.empty_like(data)  # dataと同じ大きさの空の行列を作成
        row = data.shape[0]  # dataの行数を取得
        col = data.shape[1]  # dataの列数を取得

        #for i in range(row):
        #for j in range(col):
        #data_nom[i, j] = data[i, j] / max(abs(data[i, j])) #対応する列を正規化

        for i in range(col):
            data_nom[:, i] = data[:, i] / max(abs(data[:, i]))

        return data_nom


#訓練データの読み込み
data = np.loadtxt(
    "./data/save_data.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )

#テストデータの読み込み
test = np.loadtxt(
    "./data/test_data.csv", #読み込むファイル名(例"save_data.csv")
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
x_train = data_[:, 0:2] #入力データをセット
t_train = data_[:, 2]   #正解データをセット

#テストデータのセット
x_test  = test_[:, 0:2] #入力データをセット
t_test  = test_[:, 2]   #正解データをセット


#モデルの構築
model = Sequential()
model.add(InputLayer(input_shape = (2,)))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation = 'linear')) #Denseは全結合の意味
#categorical_crossentropy : 交差エントロピー誤差
model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = ['accuracy'])

#学習
epochs = 20
batch_size = 128
history = model.fit(x_train, t_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, t_test))

#検証
score = model.evaluate(x_test, t_test, verbose=1)
print('Test loss : ', score[0])
print('Test accuracy :', score[1])

loss     = history.history['loss']
val_loss      = history.history['val_loss']

nb_epoch = len(loss)
plt.plot(range(nb_epoch), loss,     marker = '.', label = 'loss')
plt.plot(range(nb_epoch), val_loss, marker = '.', label='val_loss')
plt.legend(loc = 'best', fontsize = 10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
