# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

from common.squential import*
from common.layers import*
from csvIO import csvIO

# データを読み込む
io = csvIO()
learn_file = io.open_twoD_array('./data/learn.csv')
test_file = io.open_twoD_array('./data/test.csv')

learn_data = io.twoD_Numpy(learn_file)
test_data = io.twoD_Numpy(test_file)

model = Sequential()
model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

#学習
epochs = 20
batch_size = 128
history = model.fit(learn_data, test_data, batch_size, epochs)

epoch = 5
batch = 3

