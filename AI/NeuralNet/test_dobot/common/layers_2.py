#coding: utf-8
from collections import OrderedDict
#from functions import*
import importlib
#import weight
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

"""
def Dense(Units, activation='relu', weight_initializer='glorot_uniform', bias_initializer='zeros'):
    if (type(Units) == int):
        if (activation in dir(functions) == True):
            if (weight_initializer in dir(weight) == True):
                if (bias_initializer in dir(bias) == True):
                    return Units, activation, weight_initializer, bias_initializer

    else:
        print('error Units!')
"""

def CallFunction(function_name):
    return importlib.import_module(function_name)


def _init_params(call, initializer):
    m = CallFunction(call)  # モジュール呼び出し
    if (initializer in dir(m)):  # 関数がモジュール内にあるか確認
        return getattr(m, initializer)  # 関数呼び出し
    else:
        print('initializerが存在しません！')



class Dense:
    def __init__(self, Units=50, activation='relu', weight_initializer='he', bias_initializer='zeros'):
        self.units = Units
        self.params = {}


        # 重みの初期化
        self.__init_weight(weight_initializer)
        self.__init_bias(bias_initializer)
        print(self.params)


    def __init_weight(self, weight_initializer):
        """
        重みの初期設定

        Parameters
        ----------
        weight_initializer : 重みの標準偏差を指定
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        """
        if str(weight_initializer).lower() in ('relu', 'he'):
            weight_initializer = 'he_nomal'
        elif str(weight_initializer).lower() in ('sigmoid', 'xavier'):
            weight_initializer = 'glorot_uniform'
        method = _init_params('weight', weight_initializer)
        scale = method(self.units)
        self.params['W'] = scale * np.random.randn(10, 1)

   
    def __init_bias(self, bias_initializer):
        """
        重みの初期設定

        Parameters
        ----------
        bias_initializer : biasを指定
        """
        method = _init_params('bias', bias_initializer)
        self.params['b'] = method(self.units)




if __name__ == "__main__":
    #print(InputLayer(2))
    #print(InputLayer((2, 3)))

    dense = Dense()
    #Dense(weight_initializer='relu')
