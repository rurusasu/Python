#coding: utf-8
from collections import OrderedDict
from functions import*
import numpy as np
import sys
import os
sys.path.append(os.pardir)


def InputLayer(x):
    """
    fit時に入力データの行数を整列する関数

    """
    if (type(x) == int):
        return x
    if (type(x) == tuple):
        return x
    else:
        print('error InputLayer!')


def Dense(Units, activation='relu', weight_initializer='glorot_uniform', bias_initializer='zeros'):
    if (type(Units) == int):
        if (activation in dir(functions) == True):
            if (weight_initializer in dir(weight) == True):
                if (bias_initializer in dir(bias) == True):
                    return Units, activation, weight_initializer, bias_initializer

    else:
        print('error Units!')


if __name__ == "__main__":
    print(InputLayer(2))
    print(InputLayer((2, 3)))


