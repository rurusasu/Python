import sys, os
sys.path.append(os.pardir)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import RMSprop

from common.functions import *
import matplotlib.pyplot as plt


#モデルの構築
model = Sequential()
model.add(InputLayer(input_shape = (2,)))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation = 'liner'))

model.compile(loss = 'mean_squared_error', optimizer = 'sgd', meter)