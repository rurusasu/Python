# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

from common.squential_2 import*
from common.layers_2 import*

if __name__ == "__main__":
    model = Sequential()
    model.add(Dense(50, weight_initializer='he'))
    # print(model)
