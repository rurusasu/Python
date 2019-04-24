import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import random
from common.two_layer_net_2 import TwoLayerNet

#データを取りたい関数
def func(x):
    a=1
    b=2
    c=1
    d=1
    e=1
    f=1
    #X = random.random()
    #Y = random.random()
    #Z = a*X**2 + b*Y**2 + c*X*Y + d*X + e*Y + f
    z = a*x[0]**2 + b*x[1]**2 + c*x[0]*x[1] + d*x[0] + e*x[1] + f
    #num = np.array([X, Y, Z])
    #return num
    return z


data = np.loadtxt(
    "save_data.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )


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


#読み込んだデータを学習用にコピーする
data_1 = data_copy(data)

#標準化
data_1 = data_std(data_1)

#正規化
data_1 = data_nom(data_1)


N1 = 2 #入力の数(x, y)
N2 = 50 #第2層(第1隠れ層)のニューロン(ユニット)の数
N3 = 50 #第3層(第2隠れ層)のニューロンの数
N4 = 50 #第4層(第3隠れ層)のニューロンの数
N5 = 1  #出力層


#閾値の初期設定
#th2 = K*(np.ones((N2, 1))*0.5 - np.random.rand(N2, 1))
#th3 = K*(np.ones((N3, 1))*0.5 - np.random.rand(N3, 1))
#th4 = K*(np.ones((N4, 1))*0.5 - np.random.rand(N4, 1))
#th5 = K*(np.ones((N5, 1))*0.5 - np.random.rand(N5, 1))

#最初の重みの保存
#w_cop = w

#最初の閾値の保存
#th2_cop = th2
#th3_cop = th3
#th4_cop = th4
#th5_cop = th5

network = TwoLayerNet(input_size = N1, hidden_size = N2, output_size = N5)

iters_num = 1000 #Minibatch_sizeを使った最大学習回数
train_size = data_1.shape[0] #訓練データ内のサンプル数をdata_rowに格納
batch_size = 10 #訓練データ内のサンプル数をBatch sizeに格納
learning_rate = 0.1 #学習率

iters_list = []
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
err_sum = 0

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) #配列やリストからランダムに値を取り出す
    x_batch = data_1[batch_mask, 0:2] #x, y座標をBatchの中からランダムに10000個取り出す
    t_batch = data_1[batch_mask, 2]   #テストデータは、その時のz座標とする

    for j in range(batch_size):
        x_minibatch = x_batch[j]
        t_minibatch = t_batch[j]

        #誤差逆伝播法によって勾配を求める
        grad = network.gradient(x_minibatch, t_minibatch, batch_size, j)

        if grad == 0:
            continue

        else:
            #更新
            for key in ('W1', 'b1', 'W2', 'b2'):
                network.params[key] -= learning_rate * grad[key]

        
            loss = network.loss(x_batch, t_batch)
    
            iters_list.append(i) #x軸
            train_loss_list.append(loss) #y軸


    if j % iter_per_epoch == 0:
        train_acc = network.accuracy(x_batch, t_batch)
        train_acc_list.append(test_acc)
        print(train_acc)
   

plt.plot(iters_list, train_loss_list)
plt.xlabel("iters")
plt.ylabel("loss")
plt.show()