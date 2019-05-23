#coding: utf-8
from __future__ import unicode_literals, print_function
import numpy as np
import matplotlib.pyplot as plt



#データコピー
def data_copy(data):
    dim = data.ndim                      #受け取ったdataの次元を確認

    if dim == 1 :                        #dataが1次元(配列)のとき
        return data
    else :                               #dataが2次元(行列)のとき
        data_copy = np.empty_like(data)  #dataと同じ大きさの空の行列を作成
        col = data.shape[1]              #dataの列数を取得

        for i in range(col):
            data_copy[:, i] = data[:, i] #対応する列に値をコピー
        
        return data_copy

#標準化
def data_std(data):
    dim = data.ndim #受け取ったdataの次元を確認

    if dim == 1 :   #dataが1次元(配列)のとき
        return (data - data.mean()) / data.std()
    else :                               #dataが2次元(行列)のとき
        data_std = np.empty_like(data)   #dataと同じ大きさの空の行列を作成
        col = data.shape[1]              #dataの列数を取得

        for i in range(col):
            data_std[:, i] =(data[:, i] - data[:, i].mean()) / data[:, i].std() #対応する列を標準化
        
        return data_std


#正規化
def data_nom(data):
    dim = data.ndim #受け取ったdataの次元を確認

    if dim == 1 :   #dataが1次元(配列)のとき
        return data_1[:, 0] / max(abs(data_1[:, 0]))
    else :                               #dataが2次元(行列)のとき
        data_nom = np.empty_like(data)   #dataと同じ大きさの空の行列を作成
        col = data.shape[1]              #dataの列数を取得

        for i in range(col):
            data_nom[:, i] =data[:, i] / max(abs(data[:, i])) #対応する列を正規化
        
        return data_nom


def weight(w1, w2, w3, w4, b2, b3, b4, b5, eta, beta, x1, x2, x3, x4, x5, x5_d, s2, s3, s4, s5, w1_1, w2_1, w3_1, w4_1, eta_myu, N1, N2, N3, N4, N5, b2_1, b3_1, b4_1, b5_1, ACTIVATION, ReLU_GAIN):
    #前回の重みを保存しておく(入力層と第２層（第１隠れ層）間の重み)
    w1_tmp = w1_1
    w2_tmp = w2_1 #第２層（第１隠れ層）と第３層（第２隠れ層）間の重み
    w3_tmp = w3_1 #第３層（第２隠れ層）と第４層（第３隠れ層）間の重み
    w4_tmp = w4_1 #第４層（第３隠れ層）と第５層（出力層）間の重み
    #前回の閾値を保存しておく
    b2_tmp = b2_1
    b3_tmp = b3_1
    b4_tmp = b4_1
    b5_tmp = b5_1
    #慣性項の計算のために現在の重みを前回の重みとして保存する。
    w1_1 = w1
    w2_1 = w2
    w3_1 = w3
    w4_1 = w4
    #慣性項の計算のために、現在の閾値を前回の閾値として保存する
    b2_1 = b2
    b3_1 = b3
    b4_1 = b4
    b5_1 = b5
    delta5 = np.zeros((N5, 1))
    delta4 = np.zeros((N4, 1))
    delta3 = np.zeros((N3, 1))
    delta2 = np.zeros((N2, 1))

    #ステップ1：出力層に向かう結合係数(w4)を修正する。
    for i in range(N5):
        if ACTIVATION == 0:
            #シグモイド関数の場合
            delta5[i] = (1 / (1 + np.exp(-s5[i]))) * (1 - 1 / (1 + np.exp(-s5[i]))) * (x5[i]- x5_d[i]) #出力層におけるデルタ
        else:
            #ReLU関数の場合
            #mask = (s5 <= 0)
            #delta5 = x5 - x5_d
            #delta5[mask] = ReLU_GAIN * (x5[mask] - x5_d[mask])
            if s5[i] > 0:
                delta5[i] = (x5[i] - x5_d[i])
            else:
                delta5[i] = ReLU_GAIN * (x5[i] - x5_d[i])
        for j in range(N4):
        #w4 = w4 - eta * np.dot(delta5, x4) + eta_myu * (w4 - w4_tmp)   #慣性項ありの重み更新
        #w4 = w4 - eta * np.dot(x4.T, delta5) + eta_myu * (w4 - w4_tmp)   #慣性項ありの重み更新
            w4[j][i] = w4[j][i] - eta * delta5[i] * x4[j] + eta_myu * (w4[j][i] - w4_tmp[j][i]) #慣性項ありの重み更新
        b5[i] = b5[i] - beta * delta5[i] * x4[j] + beta_myu * (b5[i] - b5_tmp[i])               #慣性項ありの閾値の更新
    
    #その他の結合係数は入力層に向かって順次修正を行う。誤差の計算に注意！
    #ステップ２：結合係数(w3)を修正する。
    for i in range(N4):
        sigma = 0 #これが誤差となる。
        for k in range(N5):
            sigma = sigma + delta5[k] * w4[i][k] #ここが誤差の計算となっている。
        if ACTIVATION == 0:
            #シグモイド関数の場合
            delta4[i] = (1 / (1 + np.exp(-s4[i]))) * (1 - 1 / (1 + np.exp(-s4[i]))) * sigma #出力層以外のデルタ
        else:
            #ReLUの場合
            if s4[i] > 0:
                delta4[i] = sigma
            else:
                delta4[i] = ReLU_GAIN * sigma
            for j in range(N3):
                w3[j][i] = w3[j][i] - eta * delta4[i] * eta_myu * (w3[j][i] - w3_tmp[j][i]) #ユニットの重みの更新
            b4[i] = b4[i] - beta * delta4[i] * x3[j] + beta_myu * (b4[i] - b4_tmp[i])          #ユニットの閾値の更新

    #ステップ3：結合係数(w2)を修正する。
    for i in range(N3):
        sigma = 0 #これが誤差となる。
        for k in range(N4):
            sigma = sigma + delta4[k] * w3[i][k] #ここが誤差の計算となっている。
            if ACTIVATION == 0:
                #シグモイド関数の場合
                delta3[i] = (1 / (1 + np.exp(-s3[i]))) * (1 - 1 / (1 + np.exp(-s3[i]))) * sigma #出力層以外のデルタ
            else:
                #ReLUの場合
                if s3[i] > 0:
                    delta3[i] = sigma
                else:
                    delta3[i] = ReLU_GAIN * sigma
        for j in range(N2):
            w2[j][i] = w2[j][i] - eta * delta3[i] * x2[j] + eta_myu * (w2[j][i] - w2_tmp[j][i]) #ユニットの重みの更新
        b3[i] = b3[i] - beta * delta3[i] * x2[j] + beta_myu * (b3[i] - b3_tmp[i])               #ユニットの閾値の更新

    #ステップ4：結合係数(w1)を修正する。
    for i in range(N2):
        sigma = 0 #これが誤差となる。
        for k in range(N3):
            sigma = sigma + delta3[k] * w2[i][k] #ここが誤差の計算となっている。
            if ACTIVATION == 0:
                #シグモイド関数の場合
                delta2[i] = (1 / (1 + np.exp(-s2[i]))) * (1 - 1 / (1 + np.exp(-s2[i]))) * sigma #出力層以外のデルタ
            else:
                #ReLUの場合
                if s2[i] > 0:
                    delta2[i] = sigma
                else:
                    delta2[i] = ReLU_GAIN * sigma
        for j in range(N1):
            w1[j][i] = w1[j][i] - eta * delta2[i] * x1[j] + eta_myu * (w1[j][i] - w1_tmp[j][i]) #ユニットの重みの更新
        b2[i] = b2[i] - beta * delta2[i] * x1[j] + beta_myu * (b2[i] - b2_tmp[i])               #ユニットの閾値の更新

    return w1, w2, w3, w4, b2, b3, b4, b5, w1_1, w2_1, w3_1, w4_1, b2_1, b3_1, b4_1, b5_1


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
data = data_copy(data)
test = data_copy(test)

#標準化
data_1 = data_std(data)
test_1 = data_std(test)

#正規化
data_ = data_nom(data_1)
test_ = data_nom(test_1)

#訓練データのセット
x_train = data_[:, 0:2] #入力データをセット
t_train = data_[:, 2]   #正解データをセット

#テストデータのセット
x_test  = test_[:, 0:2] #入力データをセット
t_test  = test_[:, 2]   #正解データをセット


ACTIVATION = 1 #0:Sigmoid function, 1:ReLU function
ReLU_GAIN  = 0.7

if ACTIVATION == 1:
    #for ReLU
    eta      = 0.0001 #重みの学習係数(learning rate for weights),ReLUの場合、値を大きくすると誤差が減少しない。
    beta     = 0.0001 #閾値の学習係数(learning rate for bias)(活性化関数を入力軸上で微小移動)
    eta_myu  = 0.01   #慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す
    beta_myu = 0.01   #慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す

else:
    #for Sigmoid
    eta      = 0.1 #重みの学習係数(learning rate for weights),ReLUの場合、値を大きくすると誤差が減少しない。
    beta     = 0.1 #閾値の学習係数(learning rate for bias)(活性化関数を入力軸上で微小移動)
    eta_myu  = 0.1 #慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す
    beta_myu = 0.1 #慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す



#各レイヤごとの層数
N1 = 2
N2 = 50
N3 = 50
N4 = 50
N5 = 1



#重みの初期化
#w1 = np.random.randn(N1, N2) / np.sqrt(N1) 
#w2 = np.random.randn(N2, N3) / np.sqrt(N2)
#w3 = np.random.randn(N3, N4) / np.sqrt(N3)
#w4 = np.random.randn(N4, N5) / np.sqrt(N4)
K = 2
w1 = K*(np.ones((N1, N2))*0.5 - np.random.rand(N1, N2))
w2 = K*(np.ones((N2, N3))*0.5 - np.random.rand(N2, N3))
w3 = K*(np.ones((N3, N4))*0.5 - np.random.rand(N3, N4))
w4 = K*(np.ones((N4, N5))*0.5 - np.random.rand(N4, N5))

#閾値の初期化
#b2 = np.zeros(N2) 
#b3 = np.zeros(N3)
#b4 = np.zeros(N4)
#b5 = np.zeros(N5)
b2 = K*(np.ones((N2, 1))*0.5 - np.random.rand(N2, 1))
b3 = K*(np.ones((N3, 1))*0.5 - np.random.rand(N3, 1))
b4 = K*(np.ones((N4, 1))*0.5 - np.random.rand(N4, 1))
b5 = K*(np.ones((N5, 1))*0.5 - np.random.rand(N5, 1))



#最初の重みの保存
w1_1 = w1
w2_1 = w2
w3_1 = w3
w4_1 = w4

#最初の閾値の保存
b2_1 = b2
b3_1 = b3
b4_1 = b4
b5_1 = b5

#状態ベクトルの初期化
s1 = np.zeros((N1, 1))
s2 = np.zeros((N2, 1))
s3 = np.zeros((N3, 1))
s4 = np.zeros((N4, 1))
s5 = np.zeros((N5, 1))

#出力ベクトルの初期化
x1 = np.zeros((N1, 1))
x2 = np.zeros((N2, 1))
x3 = np.zeros((N3, 1))
x4 = np.zeros((N4, 1))
x5 = np.zeros((N5, 1))

Batch_size = x_train.shape[0]
Iteration_limit = 20 # epoche回数
Minibatch_size = 1000
'''
学習の進捗状況（訓練データ内の１サンプルあたりの誤差）を保存するバッファ
誤差は目標出力とNNからの出力との差
'''
Ev_buff = np.zeros((Iteration_limit, 1))
plot = Plot(0, Ev_buff[0])

for iteration in range(Iteration_limit):
    #行数からMinibatch_sizeだけランダムに値を抽出 replace(重複)
    Batch_mask = np.random.choice(Batch_size, Minibatch_size, replace = False)
    data_ = x_train[Batch_mask, 0:2]  #訓練データをセット
    t_data_ =  t_train[Batch_mask]       #全訓練データからBatch_maskだけ抽出

    #抽出された訓練データの行数と列数を取得
    Row    = data_.shape[0]
    Column = data_.shape[1]
    #標準化と正規化されたデータ保存用
    x5_buff     = np.zeros((Row, N5)) #NNからの出力（第5層のユニットからの出力）の保存先を確保
    x5_err_buff = np.zeros((Row, N5)) #NNからの出力（第5層のユニットからの出力）と訓練データとの誤差の保存先を確保
    #標準化と正規化される前のオリジナルデータ保存用
    x5_buff2 = np.zeros((Row, N5))
    x5_err_buff2 = np.zeros((Row, N5))
    for sample_counter in range(Row):
        print('学習回数：%d' %sample_counter)
        #第一層
        s1[0] = data_[sample_counter][0]
        s1[1] = data_[sample_counter][1]
        x1[0] = s1[0] #入力層ではそのまま出力される
        x1[1] = s1[1]
        #第二層
        for i in range(N2):
            s2[i] = 0
            for j in range(N1):
                s2[i] = s2[i] + w1[j][i] * x1[j] #第二層の状態ベクトルの成分を求める。
            s2[i] = s2[i] + b2[i] #状態量に閾値を加える。-> 活性化関数への入力となる。
        if ACTIVATION == 0:
            x2[i] = (1 / (1 + np.exp(-s2[i]))) - 0.5
        else:
            if s2[i] > 0:
                x2[i] = s2[i]
            else:
                x2[i] = ReLU_GAIN * s2[i]
        #第三層
        for i in range(N3):
            s3[i] = 0
            for j in range(N2):
                s3[i] = s3[i] + w2[j][i] * x2[j] #第二層の状態ベクトルの成分を求める。
            s3[i] = s3[i] + b3[i] #状態量に閾値を加える。-> 活性化関数への入力となる。
        if ACTIVATION == 0:
            x3[i] = (1 / (1 + np.exp(-s3[i]))) - 0.5
        else:
            if s3[i] > 0:
                x3[i] = s3[i]
            else:
                x3[i] = ReLU_GAIN * s3[i]
        #第四層
        for i in range(N4):
            s4[i] = 0
            for j in range(N3):
                s4[i] = s4[i] + w3[j][i] * x3[j] #第二層の状態ベクトルの成分を求める。
            s4[i] = s4[i] + b4[i] #状態量に閾値を加える。-> 活性化関数への入力となる。
        if ACTIVATION == 0:
            x4[i] = (1 / (1 + np.exp(-s4[i]))) - 0.5
        else:
            #mask = (s4 <= 0) #xが0以下ならFalseを返す(0より大きければTrue)
            #x4[mask] = ReLU_GAIN * s4[mask]
            if s4[i] > 0:
                x4[i] = s4[i]
            else:
                x4[i] = ReLU_GAIN * s4[i]
        #第五層
        for i in range(N5):
            s5[i] = 0
            for j in range(N4):
                s5[i] = s5[i] + w4[j][i] * x4[j] #第二層の状態ベクトルの成分を求める。
            s5[i] = s5[i] + b5[i] #状態量に閾値を加える。-> 活性化関数への入力となる。
            x5[i] = s5[i]

        x5_d = t_data_.reshape(-1, 1)
        for i in range(N5):
            #標準化と正規化処理されたデータによる出力と誤差
            x5_buff[sample_counter][i] = x5[i] #NNからの出力を保存しているだけ。なくてもNNの学習はできる。
            x5_err_buff[sample_counter][i] = abs(x5_d[i] - x5[i])
            #オリジナルデータとの出力誤差
            x5_buff2[sample_counter][i] = x5[i] * max(abs(data[:, 2])) * data[:, 2].std() + data[:, 2].mean()  #対応する列を正規化
            x5_err_buff2[sample_counter][i] = abs(data[sample_counter, 2] - x5_buff2[sample_counter][i])
            
        w1, w2, w3, w4, b2, b3, b4, b5, w1_1, w2_1, w3_1, w4_1, b2_1, b3_1, b4_1, b5_1 = weight(w1, w2, w3, w4, b2, b3, b4, b5, eta, beta, x1, x2, x3, x4, x5, x5_d, s2, s3, s4, s5, w1_1, w2_1, w3_1, w4_1, eta_myu, N1, N2, N3, N4, N5, b2_1, b3_1, b4_1, b5_1, ACTIVATION, ReLU_GAIN)

        err_sum = 0
    for i in range(sample_counter):
        for j in range(N5):
            err_sum = err_sum + x5_err_buff[i][j] #Minibatch_sizeで指定されたデータ分の誤差の総和
    Ev_buff[iteration] = err_sum / sample_counter #訓練データ内の１サンプルあたりの平均誤差を保存
    #if rem[iteration, 0] == 0
    plot.grah_plot(iteration+1, Ev_buff[iteration])

plot.grah_plot(iteration+1, Ev_buff[iteration])
