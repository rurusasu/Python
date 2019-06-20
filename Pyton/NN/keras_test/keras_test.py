import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt

#データを読み込む
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Mnistデータを加工する
x_train = x_train.reshape(60000, 784) #1次元配列に変換
x_test = x_test.reshape(10000, 784)

#データをfloat型に変換
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#0～255までの範囲のデータを0～1までの範囲に変更
x_train /= 255
x_test /= 255

#正解データの加工
y_train = keras.utils.to_categorical(y_train, 10) #one_hot_labelに変換
y_test  = keras.utils.to_categorical(y_test,  10)

#モデルの構築
model = Sequential()
model.add(InputLayer(input_shape = (784,)))
#model.add(Dense(50, activation = 'relu'))
#model.add(Dense(50, activation = 'relu'))
model.add(Dense(10, activation = 'softmax')) #Denseは全結合の意味
#categorical_crossentropy : 交差エントロピー誤差
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

#学習
epochs = 20
batch_size = 128
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

#検証
score = model.evaluate(x_test, y_test, verbose=1)
print()
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