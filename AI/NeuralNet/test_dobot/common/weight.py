# cording: utf-8

import numpy as np


#重みの初期値
#Xavierの一様分布
def glorot_uniform(self, input_dim, h_dim):
    """
    Initialize weights with Xavier

    Parameters
    ----------
    x : numpy.ndarray
        2次元配列

    Returns
    -------
    weight : numpy.ndarray
        2次元配列
    """
    array = x.T
    
    np.random.seed(0)
    weight = np.random.randn(input_dim, h_dim) / np.sqrt(input_dim)

    return weight


#Heの初期値
def he_nomal(self, input_dim, h_dim):
    weight = 
        np.sqrt(2) * np.random.randn(input_dim, h_dim) / np.sqrt(input_dim)

    return weight


if __name__ == "__main__":
    