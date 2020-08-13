# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

import numpy as np
from common.analysis import CenterDiffMethod


def _numerical_gradient_1d(f, x):
    # 数値解析
    grad = CenterDiffMethod(f, x)

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)  # gradの初期化

        for idx, x in enumerate(X):  # enumrate(x) インデックス番号, 要素の順に取得する
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad
