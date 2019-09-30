# cording: utf-8

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=0)

"""
def function_2(X, Y):
    return X**2 + Y**2
"""
"""
def function_2(X):
    return X[0]**2 + X[1]**2
"""

if __name__ == '__main__':
    x0 = np.arange(-3, 3, 0.1)
    x1 = np.arange(-3, 3, 0.1)
    X, Y = np.meshgrid(x0, x1) # meshgrid：配列の要素から格子列を作成

    f = function_2(np.array([X, Y]))

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, f)
    plt.show()