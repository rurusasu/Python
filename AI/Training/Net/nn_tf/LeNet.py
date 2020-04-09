import keras
import cv2
import numpy as np
import argparse
from glob import glob
from DataAugumentation import DataLoad

# GPU Config
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(
            physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

# network
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization

num_classes = 2
img_height, img_width = 32, 32

def LeNet():
    # 入力層
    inputs = Input((img_height, img_width, 3))
    # 畳み込み層
    x = Conv2D(6, (5, 5), padding='valid', activation=None, name='conv1')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Conv2D(16, (5, 5), padding='valid', activation=None, name='conv2')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Activation('sigmoid')(x)

    x = Flatten()(x)
    # 全結合層
    x = Dense(120, name='dense1', activation=None)(x)
    x = Dense(64, name='dense2', activation=None)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x, name='model')
    return model

def tarin(DirPath):
    model = LeNet()

    for layer in model.layers:
        layer.trainable =True

    model.compile(
        loss = 'categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy']
    )

    xs, ts, paths = DataLoad.Data_load(DirPath, Flamework='Keras')

    # training
    x, t = DataLoad.MiniBatch(xs, ts, mb=8, mbi=0)

# 再現性確保のために乱数シード値を固定（数値はなんでも良い）
tf.compat.v1.set_random_seed(12345)

# 入力データ
# MNISTのone-hot表現での読み込み

# 入力画像
x = tf.compat.v1.placeholder(tf.float32, name = 'x')
# サイズ変更
x_1 = tf.reshape(x, [-1, 28, 28, 1])
# 畳み込み
# ランダムカーネル
k_0 = tf.Variable(tf.random.truncated_normal([4, 4, 1, 10], mean=0.0, stddev=0.1))
# 畳み込み
x_2 = 
#W = tf.Variable(tf.random_uniform_initializer)

if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    from PIL import Image

    # MNIST_dataディレクトリにMNISTデータをダウンロードして読み込む
    mnist = input_data.read_data_sets('MNIST_data/')
    # 1枚目を28x28ピクセルの行列に変換
    image_matrix = tf.reshpe(mnist.train.images[0], [28, 28])
