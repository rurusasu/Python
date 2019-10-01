# cording: utf-8

import numpy as np
import matplotlib.pylab as plt


#-------------------------------------------------
# 常微分方程式の解法(gradient.pyに使用)
#-------------------------------------------------
def CenterDiffMethod(f, x):
    """
    Function of center difference method

    Parameters
    ----------
    x : 1D list
        差分計算用の1次元配列
        
    Returns
    -------
    grad : 1D list
        勾配の計算結果（1次元配列）

    """
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