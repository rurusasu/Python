import numpy as np

# ステップ関数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU関数（ランプ関数）
def relu(x):
    return np.maximum(0, x)


#出力層
#恒等関数(回帰問題で使用)
def LinerFunction(x):
    return x

# tanh関数
def tanh(x):
    return np.tanh(x)
    
# logistic関数
def logistic(x):
    y = 1 / (1 + np.exp(-x))
    return y

#ソフトマックス関数(分類問題で使用)
def softmax(x):
    if x.ndim == 2:
        x = x.T
        y = x - np.max(x, axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


#損失関数
#2乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2, axis=1, keepdims=True)


#交差エントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    #教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
