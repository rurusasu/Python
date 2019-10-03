#coding: utf-8

from collections import OrderedDict
#from functions import*
from functions import _CallFunction
import numpy as np


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
def _Call(function_name):
    return importlib.import_module(function_name)


def _CallFunction(module, function):
    m = _Call(module)  # モジュール呼び出し
    if (function in dir(m)):  # 関数がモジュール内にあるか確認
        return getattr(m, function)  # 関数呼び出し
    else:
        print('functionが存在しません！')
"""


class Dense:
    def __init__(self, Units, activation='relu', weight_initializer='he', bias_initializer='zeros'):
        if (Units != None and type(Units) == int):
            self.units = Units
        self.initializer  = {} 
        self.initializer['w'] = weight_initializer
        self.initializer['b'] = bias_initializer
        self.params = {}      


    def _initParams(self, forward_node):
        # 重みの初期化
        #self.__init_weight(self.initializer['w'])
        self.__init_weight(self.initializer['w'])
        self.__init_bias(self.initializer['b'])

    """
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
        method = _CallFunction('weight', weight_initializer)
        scale = method(self.units)
        self.params['W'] = scale * np.random.randn(10, 1)

   
    def __init_bias(self, bias_initializer):
        """
        重みの初期設定

        Parameters
        ----------
        bias_initializer : biasを指定
        """
        method = _CallFunction('bias', bias_initializer)
        self.params['b'] = method(self.units)




if __name__ == "__main__":
    #print(InputLayer(2))
    #print(InputLayer((2, 3)))
    
    dense = Dense(50)
    #Dense(weight_initializer='relu')
