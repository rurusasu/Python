# cording: utf-8

import numpy as np


#重みの初期値
#Xavierの一様分布
def glorot_uniform(node_num):
    """
    Initialize weights with Xavier

    Parameters
    ----------
    node_num : int
        前層のnodeの個数

    Returns
    -------
    weight : float
        重みの初期値
    """
    return np.sqrt(1.0 / node_num)


#Heの初期値
def he_nomal(node_num):
    return np.sqrt(2.0 / node_num)


if __name__ == "__main__":
    print(glorot_uniform(50))
    print(he_nomal(50))