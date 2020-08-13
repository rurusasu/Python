import numpy as np
import matplotlib.pyplot as plt
import time


def perceptron(w_vec, x_vec, label):
    low=0.5 # 学習係数
    if (np.dot(w_vec, x_vec) * label >= 1):
        w_vec_new = w_vec + label*low*x_vec
        return w_vec_new
    else:
        return w_vec


#-----------------------
# 調整用パラメータ
#-----------------------
class_1 = 5  # class_1の(x, y)座標の倍率(正の整数)
class_2 = -2  # class_2の(x, y)座標の倍率(負の整数)
eta = 0.001  # 学習率(0 <= eta <= 1)
loop = 100   # 反復回数

if __name__ == "__main__":
    train_num = 100  # 学習データ数

    # class1の学習データ
    x1_1 = np.random.rand(int(train_num/2))*5 + class_1  # x成分
    x1_2 = np.random.rand(int(train_num/2))*5 + class_1  # y成分
    label_x1 = np.ones(int(train_num/2))  # ラベル(すべて1)

    # class2の学習データ
    x2_1 = (np.random.rand(int(train_num/2))*5 + 1) * class_2
    x2_2 = (np.random.rand(int(train_num/2))*5 + 1) * class_2
    label_x2 = np.ones(int(train_num/2)) * -1  # ラベル(すべて-1)

    x0 = np.ones(int(train_num/2))  # x0は常に1(バイアス)
    x1 = np.c_[x0, x1_1, x1_2]
    x2 = np.c_[x0, x2_1, x2_2]

    x_vecs = np.r_[x1, x2]
    labels = np.r_[label_x1, label_x2]

    w_vec = np.array([2, -1, 3], dtype='float64')  # 初期の重みベクトル 適当に決める

    start = time.time()
    for i in range(loop):
        for x_vec, label in zip(x_vecs, labels):
            w_vec = perceptron(w_vec, x_vec, label)

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print(w_vec)

    # グラフの体裁を整える
    plt.rcParams['font.family'] = 'sans-serif'  # 使用するフォント
    plt.rcParams['xtick.direction'] = 'in' # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in' # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
    plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
    plt.rcParams['font.size'] = 11  # フォントの大きさ
    plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ

    plt.scatter(x1[:,1], x1[:,2], c = 'red', marker='.', label='group1')
    plt.scatter(x2[:,1], x2[:,2] ,c = 'blue', marker='^', label='group2')

    # 分離境界線
    x_fig = np.array(range(-8, 8))
    y_fig = -(w_vec[1]/w_vec[2])*x_fig - (w_vec[0]/w_vec[2])

    plt.plot(x_fig, y_fig, color='black')

    #plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)

    # グラフ右と上の軸を消す
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.xlabel('X')
    plt.ylabel('Xn')

    plt.show()
