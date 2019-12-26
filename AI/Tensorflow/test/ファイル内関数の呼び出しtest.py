import sys, os
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
import importlib


def _Call(function_name):
    return importlib.import_module(function_name)

def _CallFunction(module, function):
    """
    Call function with module

    Parameters
    ----------
    module : str
        呼び出したいmodule名
    function : str
        module内の関数名
    
    Return
    ------
    method : method
        実行可能な関数
    """
    m = _Call(module)  # モジュール呼び出し
    if (function in dir(m)):  # 関数がモジュール内にあるか確認
        return getattr(m, function)  # 関数呼び出し
    else:
        print('functionが存在しません！')

method = _CallFunction('common.functions', 'Relu')

sess = tf.Session()
print(sess.run(method([-3., 3., 10.])))