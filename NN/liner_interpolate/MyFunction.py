#!/usr/bin/env python
# coding: utf-8

#
import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted



def load_data(path):
    def get_image(label_path):
        x, t = [], []
        label_path = natsorted(label_path)
        for (i, j) in zip(label_path, range(len(label_path))):
            for k in glob.glob(i + "/*"):
                img = plt.imread(k)
                x.append(img)
                t.append(j)
        x = np.array(x)
        t = np.array(t)
        t = t.reshape(-1, 1)
        
        a = np.random.permutation(len(x))
        x = x[a]
        t = t[a]
        
        return x, t
       
    # trainingデータの取得
    label_path = glob.glob(path + "/training/*")
    x_train , t_train = get_image(label_path)

    # testデータの取得
    label_path = glob.glob(path + "/test/*")
    x_test , t_test = get_image(label_path)   

    return (x_train, t_train), (x_test, t_test)

