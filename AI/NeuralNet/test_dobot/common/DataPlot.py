# cording: utf-8

import numpy as np
import matplotlib.pylab as plt


def PlanePlot(X, Y):
    plt.plot(X, Y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def twoD_ArrayQuiver(X, Y, Z):
    """
    Represent a 2D array with arrows
    
    Parameters
    ----------
    X : numpy.ndarray
        メッシュのX座標
    Y : numpy.ndarray
        メッシュのY座標
    Z : numpy.ndarray
        メッシュのZ座標
    """

    plt.figure()
    plt.quiver(X, Y, -Z[0], -Z[1], angles='xy', color='#666666')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()


if __name__ == "__main__":
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)

    # 平面に計算結果をプロットする
    #Y = 0.01*x0**2 + 0.1*x0
    #PlanePlot(x0, Y)

    X, Y = np.meshgrid(x0, x1)  # meshgrid：配列の要素から格子列を生成する

    X = X.flatten()  # flatten：配列を1次元に変換する
    Y = Y.flatten()

    x = np.array([X, Y])
    Z = np.sum(x**2, axis=0)

    twoD_ArrayQuiver(X, Y, Z)
