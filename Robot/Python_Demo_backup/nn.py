# cording: utf-8

import sys,os
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from common.sequential import Sequential
from common.layers import Input, Dense


def nn(x, t, batch_size, epochs):
    model = Sequential()
    model.add(Input(input_shape=x.shape[1]))
    #model.add(Dense(50, activation='relu'))
    #model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
    model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
    model.add(Dense(t.shape[1],  activation='liner'))
    model.compile(loss='mean_squared_error')

    history=model.fit(x, t, batch_size=batch_size, epochs=epochs)

    # lossグラフ
    loss = history['loss_ave']
    nb_epoch = len(loss)
    plt.plot(range(nb_epoch), loss, marker = '.', label = 'loss')
    plt.legend(loc = 'best', fontsize = 10)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == "__main__":
    """
    from common.csvIO import csvIO
    io = csvIO() 
    x = io.open_twoD_array('./data/learn.csv')
    t = io.open_twoD_array('./data/test.csv')

    x = io.twoD_FroatToStr(x, digit=0.01)
    t = io.twoD_FroatToStr(t, digit=0.01)
    x = io.twoD_Numpy(x)
    t = io.twoD_Numpy(t)
    print('x=', x)
    print('t=', t)
    """
    import numpy as np
    from common.functions import*

    #訓練データの読み込み
    x = np.loadtxt(
        "./data/learn.csv", #読み込むファイル名(例"save_data.csv")
        dtype=float,     #データのtype
        delimiter=",",   #区切り文字の指定
        ndmin=2          #配列の最低次元
        )

    #テストデータの読み込み
    t = np.loadtxt(
        "./data/test.csv", #読み込むファイル名(例"save_data.csv")
        dtype=float,     #データのtype
        delimiter=",",   #区切り文字の指定
        ndmin=2          #配列の最低次元
        )
   
    #読み込んだデータを学習用にコピーする
    #data_1 = data_copy(x)
    #test_1 = data_copy(t)

    """
    #標準化
    x = data_std(x)
    t = data_std(t)
    """
    
    #正規化
    data_ = data_nom(x)
    test_ = data_nom(t)
    

    #訓練データのセット
    x = data_  # 学習データをセット
    t = test_  # 教師データをセット
 

    epochs=1000
    batch_size=128

    nn(x, t, batch_size=batch_size, epochs=epochs)
    
