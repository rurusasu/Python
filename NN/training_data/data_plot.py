from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

#行列の範囲の設定
row_renge = 3 #列数
x_renge_max = 100  #x軸の最大値
x_renge_min = -100 #x軸の最小値
y_renge_max = 100  #y軸の最大値
y_renge_min = -100 #y軸の最小値
delimiter = 1 #値の間隔


#データを取りたい関数
def func(X, Y):
    a=1
    b=2
    c=1
    d=1
    e=1
    f=1
    return a*X**2 + b*Y**2 + c*X*Y + d*X + e*Y + f

#ファイルへの書き込みを行う関数
def write(file_name, numpy_data):
    #data_save
    np.savetxt(
        file_name,       #書き込むファイル名の指定(例:"save_data.csv")
        numpy_data,      #書き込むデータ 
        fmt="%.5f",      #少数で書き込み
        delimiter=",",   #区切り文字の指定
        newline="\n"     #改行文字の指定
        )

#3dグラフを表示する関数
def plot3d(X, Y):
    X, Y = np.meshgrid(X, Y)  #2次元のメッシュを作る
    Z = func(X, Y)

    #print(X)
    #print(type(X))
    #print(Z)

    fig = plt.figure()  #figureで2次元の図を作成する
    ax = Axes3D(fig) #その後、Axes3D関数で3次元にする

    #軸ラベルの設定
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    #グラフの描写
    ax.plot_wireframe(X, Y, Z)
    plt.show()


def func_data(x_renge_max, xrenge_min, y_renge_max, y_renge_min, delimiter, row_renge):
    #初期設定
    data = np.arange(1, row_renge+1, delimiter)
    x = np.arange(x_renge_min, x_renge_max+1, delimiter) #xの配列を作成
    y = np.arange(y_renge_min, y_renge_max+1, delimiter) #yの配列を作成
    counter = 0

    for i in x:
        for j in y:
            z = func(i, j)
            num = np.array([i, j, z])
            #print(num)
            if (counter == 0):
                data = num
            else:
                data = np.vstack((data, num))
            counter = counter + 1
    
    #dataをグラフとして描写
    plot3d(x, y)

    #データをファイルに書き込み
    write("save_data.csv", data)



func_data(x_renge_max, x_renge_min, y_renge_max, y_renge_min, delimiter, row_renge)
