# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

from common.squential_3 import*
from common.layers import*
from common.csvIO import csvIO
import matplotlib.pyplot as plt


# データを読み込む
io = csvIO()
learn_file = io.open_twoD_array('./data/learn.csv')
test_file = io.open_twoD_array('./data/test.csv')

learn_data = io.twoD_Numpy(learn_file)
test_data = io.twoD_Numpy(test_file)

model = Sequential()
model.add(InputLayer(input_shape=(3,))) #3, 128
model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
model.add(Dense(3, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

#学習
epochs = 20
batch_size = 128
history = model.fit(learn_data, test_data, batch_size, epochs)
print(history['loss'])

loss = history['loss']

nb_epoch = len(loss)
plt.plot(range(nb_epoch), loss, marker='.', label='loss')
plt.legend(loc= 'best', fontsize = 10)
plt.grid(False)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()