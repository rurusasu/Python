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
    data = {}
    data['x']=x
    data['t']=t
    # データの仕分け
    #data['x'], data['t'], data['val'] = __sorting__(x)
    # データの前処理
    for i in data.keys():
        data[i] = __feature__(data[i], feature) # 標準化と正規化
        data[i] = __shuffle__(data[i], 0)       # データシャッフル
    # データのシャッフル
    #x, t = __shuffle__(x, t, 0)
    #data['x'], data['t'] = __shuffle__(data['x'], data['t'], 0)

    model = Sequential()
    model.add(Input(input_shape=x.shape[1]))
    #model.add(Dense(50, activation='relu', weight_initializer='relu'))
    #model.add(Dense(50, activation='relu', weight_initializer='relu'))
    model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
    model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
    #model.add(Dense(t.shape[1],  activation='softmax'))
    #model.compile(loss='cross_entropy_error')
    model.add(Dense(t.shape[1], activation = 'liner'))
    model.compile(loss='mean_squared_error')


    #history=model.fit(x, t, batch_size=batch_size, epochs=epochs, validation=validation)
    history = model.fit(data['x'], data['t'], batch_size=batch_size,
                        epochs=epochs, validation=validation)

    # lossグラフ
    loss = history['loss_ave']
    val_loss = history['val_loss']
    nb_epoch = len(loss)
    plt.plot(range(nb_epoch), loss, marker = '.', label = 'loss')
    plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
    plt.legend(loc = 'best', fontsize = 10)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def __feature__(x, feature):
    """
    データの前処理を行う関数

    Parameters
    ----------
    x : ndarray
        入力データ
    feature : int
        前処理の種類を選択
    
    Returns
    -------
    x : ndarray
        変換後のデータ
    """
    if (feature != None):
        # 標準化
        if (feature == 0 or feature == 2):
            x = data_std(x)
        # 正規化
        if (feature == 1 or feature == 2):
            x = data_nom(x)
        return x
    else:
        return x

def __shuffle__(x, seed=0):
    """
    入力配列を行に関してシャッフルする関数
    """
    rand_state = np.random.RandomState(seed)
    rand_state.shuffle(x)
    #rand_state.seed(seed)
    #rand_state.shuffle(t)
    #x = x.random.shuffle(x)
    #t = t.random.shuffle(t)
    return x

def __sorting__(x, learn_size=0.6, test_size=0.2):
    lrnMask = int(learn_size*x.shape[0])
    tstMask = int(test_size*x.shape[0]) + lrnMask
    assert tstMask <= x.shape[0], '必要データ数[{0}], 現在のデータ数[{1}]'.format(tstMask, x.shape[0])
    
    # 入力
    lrn = x[0:lrnMask]
    tst = x[lrnMask:tstMask]
    if (x.shape[0] - tstMask >= 100):
        val = x[tstMask:x.shape[0]]
        return lrn, tst, val
    else:
        val = None
        return lrn, tst, val


if __name__ == "__main__":
    import numpy as np
    from common.functions import*

    """
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

    #標準化
    x = data_std(x)
    t = data_std(t)
    x_val = data_std(x_val)
    t_val = data_std(t_val)

    #正規化
    x = data_nom(x)
    t = data_nom(t)
    x_test = data_nom(x_val)
    t_test = data_nom(t_val)
    """
    
    #データを読み込む
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    #データをfloat型に変換
    x = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #0～255までの範囲のデータを0～1までの範囲に変更
    x /= 255
    x_test /= 255

    #Mnistデータを加工する
    x = x_train.reshape(60000, 784)  # 1次元配列に変換
    x_test = x_test.reshape(10000, 784)

    #正解データの加工
    t = keras.utils.to_categorical(t_train, 10)  # one_hot_labelに変換
    t_test = keras.utils.to_categorical(t_test,  10)
    
    epochs=100
    batch_size=128

    nn(x, t, batch_size=batch_size, epochs=epochs, feature=None, validation=(x_test, t_test))
    
