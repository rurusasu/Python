# cording: utf-8

import numpy as np


# biasの初期値
def zeros(node_num):
    return np.zeros(node_num)


if __name__ == "__main__":
    print(zeros(100))