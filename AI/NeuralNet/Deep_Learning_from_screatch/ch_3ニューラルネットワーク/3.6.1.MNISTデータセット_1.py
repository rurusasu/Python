import sys
from pathlib import Path
from dataset.mnist import load_mnist

sys.path.append(Path.cwd().parent)

#最初の呼び出しは数分待ちます...
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

#それぞれのデータの形状を出力
print(x_train.shape) #(60000, 784)
print(t_train.shape) #(60000,)
print(x_test.shape) #(10000, 784)
print(t_test.shape) #(10000,)
