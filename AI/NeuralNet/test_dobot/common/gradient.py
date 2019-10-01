# cording: utf-8

import numpy as np


def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 値を元に戻す

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)  # gradの初期化

        for idx, x in enumerate(X):  # enumrate(x) インデックス番号, 要素の順に取得する
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad


if __name__ == "__main__":
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    
    X, Y = np.meshgrid(x0, x1) # meshgrid：配列の要素から格子列を生成する

    X = X.flatten()  # flatten：配列を1次元に変換する
    Y = Y.flatten()

    x = np.array([X, Y])
    Z = np.sum(x**2, axis=1)

    grad = numerical_gradient_2d(Z, np.array([X, Y]) )

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles='xy', color='#666666')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
