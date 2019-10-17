import sys
from pathlib import Path
sys.path.append(Path.cwd().parent.resolve())  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import _convert_numpy
from common.functions import sigmoid, softmax


def _change_one_hot_label_1(X):
    T = np.zeros((X.shape[0], 10))
    for i in range(X.shape[0]):
        T[i, X[i]] = 1
    return T

dataset = _convert_numpy()
dataset["train_label"]=_change_one_hot_label_1(dataset["train_label"])

print(dataset)
tra_i = dataset["train_img"]
tra_l = dataset["train_label"]
tes_i = dataset["test_img"]
tes_l = dataset["test_label"]

 #uint8:8bitの符号なし整数
print(tra_l, tra_l.shape, tra_l.shape[0], tra_l.dtype)
print(tes_l, tes_l.shape, tes_l.shape[0], tes_l.dtype)
print(tra_i, tra_i.shape, tra_i.shape[0], tra_i.dtype)
print(tes_i, tes_i.shape, tes_i.shape[0], tes_i.dtype)

t = np.zeros((tra_l.shape[0], 10))
print(t, t.shape, t.shape[0], t.dtype, t.ndim)

x = range(tra_l.shape[0])
for i in x:
    t[i, tra_l[i]]=1

y = t[0, tra_l[3]]
print(t[0, ])
z = tra_l[3]
print(z)

#enumerate関数:配列の要素と同時にカウント番号を取得できる
l = ['Alice', 'Bob', 'Charlie']
for i, name in enumerate(l):
    print(i, name)

#--------file_path------------------------
print(Path('__file__'))
print(Path('__file__').parent)
print(Path('__file__').resolve())
print(Path('__file__').parent.resolve())

print(Path('..'))
print(Path('..').parent)
print(Path('..').resolve())
print(Path('..').parent.resolve())

print(os.path.dirname(os.path.abspath('__file__')))
print(Path.cwd())
print(os.getcwd())

print(Path('dir_path6/../minst.pkl').resolve())

#親ディレクトリのPath取得
print(sys.path.append(os.pardir))
print(os.pardir)
print(Path.cwd().parent)
print(sys.path.append(Path('__file__').parent))
print(sys.path.append(Path.cwd().parent))
