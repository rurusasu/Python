# cording: utf-8
import keras
from keras.datasets import mnist
import sys,os
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from common.sequential import Sequential
from common.layers import Input, Dense
from common.functions import*


def nn(x, t, batch_size, epochs, feature=None, validation=None):
    """
    簡単なニューラルネットワークのモデルを作成する関数

    Parameters
    ----------
    x : ndarray
        学習用データ
    t : ndarray
        教師データ
    batch_size : int
        バッチサイズ
    eopchs : int
        エポック数
    feature : int
        Feature Scalingの選択
    """       
    
    model = Sequential()
    model.add(Input(input_shape=x.shape[1]))
    model.add(Dense(50, activation='relu', weight_initializer='relu'))
    model.add(Dense(50, activation='relu', weight_initializer='relu'))
    #model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
    #model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
    #model.add(Dense(t.shape[1],  activation='softmax'))
    #model.compile(loss='cross_entropy_error')
    model.add(Dense(t.shape[1], activation = 'liner'))
    model.compile(loss='mean_squared_error')


    #history=model.fit(x, t, batch_size=batch_size, epochs=epochs, validation=validation)
    history = model.fit(x, t, batch_size=batch_size,
                        epochs=epochs, validation=validation)

    # lossグラフ
    loss = history['loss_ave']
    val_loss = history['val_loss']
    train_acc = history['train_acc']

    nb_epoch = len(loss)
    plt.plot(range(nb_epoch), loss, marker = '.', label = 'loss')
    plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
    plt.plot(range(nb_epoch), train_acc, marker='.', label='train_acc')
    plt.legend(loc = 'best', fontsize = 10)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == "__main__":
    import numpy as np
    from common.functions import*

    
    #訓練データの読み込み
    x = np.loadtxt(
        "./data/learn_1.csv", #読み込むファイル名(例"save_data.csv")
        dtype=float,     #データのtype
        delimiter=",",   #区切り文字の指定
        ndmin=2          #配列の最低次元
        )

    #テストデータの読み込み
    t = np.loadtxt(
        "./data/test_1.csv", #読み込むファイル名(例"save_data.csv")
        dtype=float,     #データのtype
        delimiter=",",   #区切り文字の指定
        ndmin=2          #配列の最低次元
        )
    """
    x_val = np.loadtxt(
        "./data/val_l.csv",  # 読み込むファイル名(例"save_data.csv")
        dtype=float,  # データのtype
        delimiter=",",  # 区切り文字の指定
        ndmin=2  # 配列の最低次元
    )

    t_val = np.loadtxt(
        "./data/val_t.csv",  # 読み込むファイル名(例"save_data.csv")
        dtype=float,  # データのtype
        delimiter=",",  # 区切り文字の指定
        ndmin=2  # 配列の最低次元
    )
    """
    """
    #標準化
    x = data_std(x)
    t = data_std(t)
    #x_val = data_std(x_val)
    #t_val = data_std(t_val)

    #正規化
    x = data_nom(x)
    t = data_nom(t)
    #x_test = data_nom(x_val)
    #t_test = data_nom(t_val)

    # 標準化と正規化
    x = Datafeature(x, 2) 
    t = Datafeature(t, 2)
    # クロスバリエーション
    x_train, x_test, t_train, t_test, x_val, t_val = train_test_splint(x, t, 1)
    """
    
    #データを読み込む
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    #データをfloat型に変換
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #0～255までの範囲のデータを0～1までの範囲に変更
    x_train /= 255
    x_test /= 255

    #Mnistデータを加工する
    x_train = x_train.reshape(60000, 784)  # 1次元配列に変換
    x_test = x_test.reshape(10000, 784)

    #正解データの加工
    t_train = keras.utils.to_categorical(t_train, 10)  # one_hot_labelに変換
    t_test = keras.utils.to_categorical(t_test,  10)
    


    epochs=100
    batch_size=128

    #nn(x_train, t_train, batch_size=batch_size, epochs=epochs, feature=2, validation=(x_val, t_val))
    nn(x_train, t_train, batch_size=batch_size,
       epochs=epochs, feature=2)
    
