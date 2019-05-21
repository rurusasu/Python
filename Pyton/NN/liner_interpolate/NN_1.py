import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt  
from common.two_layer_net_2 import TwoLayerNet

#データをグラフ化する
class data_plot:
    def __init__():
        x_renge = np.empty()
        y_renge = np.empty()

    def Plot(self, epoch, loss):
        x_range = x_range.append(epoch)
        y_range = y_range.append(loss)

        plt.plot(x, y)
        plt.show()

        return 0

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


data = np.loadtxt(
    "save_data.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )


data_1 = np.empty_like(data)

#読み込んだデータを学習用にコピーする
data_1 = data_copy(data)

#標準化
#data_1[:, 0] = (data_1[:, 0] - data_1[:, 0].mean()) / data_1[:, 0].std()
#data_1[:, 1] = (data_1[:, 1] - data_1[:, 1].mean()) / data_1[:, 1].std()
#data_1[:, 2] = (data_1[:, 2] - data_1[:, 2].mean()) / data_1[:, 2].std()
data_1 = data_std(data_1)


#正規化
#data_1[:, 0] = data_1[:, 0] / max(abs(data_1[:, 0]))
#data_1[:, 1] = data_1[:, 1] / max(abs(data_1[:, 1]))
#data_1[:, 2] = data_1[:, 2] / max(abs(data_1[:, 2]))
data_1 = data_nom(data_1)


N1 = 2 #入力の数(x, y)
N2 = 50 #第2層(第1隠れ層)のニューロン(ユニット)の数
N3 = 50 #第3層(第2隠れ層)のニューロンの数
N4 = 50 #第4層(第3隠れ層)のニューロンの数
N5 = 1  #出力層

nuro = np.array([N1, N2, N3, N4, N5])
print(nuro.shape)
#重みの初期設定
K = 2
#w1 = K*(np.ones((N1, N2))*0.5 - np.random.rand(N1, N2))
#w2 = K*(np.ones((N2, N3))*0.5 - np.random.rand(N2, N3))
#w3 = K*(np.ones((N3, N4))*0.5 - np.random.rand(N3, N4))
#w4 = K*(np.ones((N4, N5))*0.5 - np.random.rand(N4, N5))


#閾値の初期設定
th2 = K*(np.ones((N2, 1))*0.5 - np.random.rand(N2, 1))
th3 = K*(np.ones((N3, 1))*0.5 - np.random.rand(N3, 1))
th4 = K*(np.ones((N4, 1))*0.5 - np.random.rand(N4, 1))
th5 = K*(np.ones((N5, 1))*0.5 - np.random.rand(N5, 1))

#最初の重みの保存
#w_cop = w

#最初の閾値の保存
th2_cop = th2
th3_cop = th3
th4_cop = th4
th5_cop = th5

data_tmp = data_1
data_1 = data

network = TwoLayerNet(input_size = 2, hidden_size = 50, output_size = 1)

DataRow_size = data_1.shape[0] #訓練データ内のサンプル数をBatch sizeに格納
Batch_size = 10000 #Dataから抽出されるサイズ
MiniBatch_size = 1 #Batchから抽出されるサイズ
Iteration_limit = 1000000 #Minibatch_sizeを使った最大学習回数
learning_rate = 0.01 #学習率

train_loss_list = []
train_acc_list = []
test_acc_list = []





iter_per_epoch = max(data_1.shape[0] / Batch_size, 1)
epoch = 1

for i in range(Iteration_limit):
    for j in range(Batch_size):
        batch_mask = np.random.choice(DataRow_size, Batch_size)
        for k in range(MiniBatch_size):
            minibatch_mask = np.random.choice(batch_mask, MiniBatch_size)
            x_batch = data_1[minibatch_mask, 0:2] #x, y座標をBatchの中からランダムに10000個取り出す
            t_batch = data_1[minibatch_mask, 2]   #テストデータは、その時のz座標とする

            #誤差逆伝播法によって勾配を求める
            grad = network.gradient(x_batch, t_batch)

            #更新
            for key in ('W1', 'b1', 'W2', 'b2'):
                network.params[key] -= learning_rate * grad[key]

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)

        plt.plot(j, train_loss_list)
        plt.show()
        #if i % iter_per_epoch == 0:
            #data_plot.Plot(epoch, loss)
            #epoch = epoch + 1