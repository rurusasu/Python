from pathlib import Path
import sys, pickle
sys.path.append(Path.cwd().parent)
import numpy as np
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

p = Path(str(Path.cwd().resolve()) + '\sample_weight\sample_weight.pkl')


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
        
    return x_test, t_test


def init_network():
    with open (p, 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    #最も確率の高い要素のインデックスを取得
    p = np.argmax(y)
    if p ==t[i]:
        accuracy_cnt += 1

print("Accuracy_cnt:" + str(float(accuracy_cnt) / len(x)))


print(x.shape)
print(t.shape)
print(x[0].shape)
W1, W2, W3 = network['W1'], network['W2'], network['W3']
print(W1.shape)
print(W2.shape)
print(W3.shape)
