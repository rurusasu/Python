import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import random
from nn_test.two_layer_net import TwoLayerNet

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
    return z


x = np.array([2.3, 3.4])
z = func(x)

N1 = 2 #入力の数(x, y)
N2 = 50 #第2層(第1隠れ層)のニューロン(ユニット)の数
N3 = 50 #第3層(第2隠れ層)のニューロンの数
N4 = 50 #第4層(第3隠れ層)のニューロンの数
N5 = 1  #出力層

iters_num = 10000
learning_rate = 0.1

#グラフの軸
iters_list = []
train_loss_list = []

#iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size = N1, hidden_size = N2, output_size = N5)


for i in range(iters_num):
    #batch_mask = np.random.choice(train_size, batch_size)
    #x_batch = data_1[batch_mask]
    #t_batch = data_1[batch_mask]

    #誤差逆伝播法によって勾配を求める
    #grad = network.gradient(x_batch, t_batch)
    grad = network.gradient(x, z)

    #更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    #loss = network.loss(x_batch, t_batch)
    loss = network.loss(x, z)
    train_loss_list.append(loss)

    iters_list.append(i)


plt.plot(iters_list, train_loss_list)
plt.xlabel("iters")
plt.ylabel("loss")
plt.show()
    
        


    