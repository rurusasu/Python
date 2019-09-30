# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt
from csvIO import csvIO



#データを読み込む
io = csvIO()
learn_file = io.open_twoD_array('./data/learn.csv')
test_file  = io.open_twoD_array('./data/test.csv')

learn_data = io.twoD_Numpy(learn_file)
test_data  = io.twoD_Numpy(test_file)


#モデルの構築
model = Sequential()
model.add(InputLayer(input_shape = (3,)))
#model.add(Dense(50, activation = 'relu'))
#model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation='softmax'))
model.add(Dense(50, activation='softmax'))
model.add(Dense(3, activation='linear'))
#categorical_crossentropy : 交差エントロピー誤差
model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = ['accuracy'])

#学習
epochs = 20
batch_size = 128
history = model.fit(learn_data, test_data, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(learn_data, test_data))


#検証
score = model.evaluate(learn_data, test_data, verbose=1)
print('Test loss : ', score[0])
print('Test accuracy :', score[1])

loss     = history.history['loss']
val_loss      = history.history['val_loss']

nb_epoch = len(loss)
plt.plot(range(nb_epoch), loss,     marker = '.', label = 'loss')
plt.plot(range(nb_epoch), val_loss, marker = '.', label='val_loss')
plt.legend(loc = 'best', fontsize = 10)
plt.grid(False)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

