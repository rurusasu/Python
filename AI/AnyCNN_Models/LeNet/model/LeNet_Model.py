# 必要なパッケージをインポートする
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization

import cv2
import numpy as np
from glob import glob
# 画像を表示するためにmatplotlibをインポートする
import matplotlib.pyplot as plt
import DataLoad_0602 as Load

# LeNetを作成する
# Network


def LeNet(img_width, img_height, num_classes):
    inputs = Input((img_height, img_width, 3))
    x = Conv2D(6, (5, 5), padding='valid',
               activation=None, name='conv1')(inputs)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Conv2D(16, (5, 5), padding='valid', activation=None, name='conv2')(x)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = Activation('sigmoid')(x)

    x = Flatten()(x)
    x = Dense(120, name='dense1', activation=None)(x)
    x = Dense(64, name='dense2', activation=None)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x, name='model')
    return model


# training
def train(DirPath, img_size, cls_label):
    model = LeNet(img_size[0], img_size[1], len(cls_label))

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    #xs, ts, paths, img_read_err, err_path = data_load(DirPath, img_size[0], img_size[1], cls_label)
    load = Load.DataLoad(DirPath, 'jpg', size=(img_size[0], img_size[1]))
    xs, ts = load.data_load()

    # training
    mb = 100
    mbi = 0
    count = 1
    loss_ls = []

    loss_ave = 0
    loss_AveMin = 1
    loss_AveCnt = 0
    acc = 0
    # loss_AveLs = []

    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)

    # for i in range(500):
    while True:
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            mbi = mb - (len(xs) - mbi)

            # loss_ls配列に値が入っているか判定
            if loss_ls:
                loss_ave = np.average(loss_ls)
            else:
                loss_ave = 0

            if loss_AveMin > loss_ave:
                loss_AveMin = loss_ave
                loss_AveCnt = 0
            else:
                loss_AveCnt += 1
            loss_ls = []
            print('iter >>', count, ',loss_ave >>',
                  loss_ave, 'accuracy >>', acc)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = xs[mb_ind]
        t = ts[mb_ind]

        loss, acc = model.train_on_batch(x=x, y=t)
        # print('iter >>', count, ',loss >>', loss, 'accuracy >>', acc)

        loss_ls.append(loss)
        count += 1

        if loss_AveCnt == 50:
            break

    model.save('LeNet.h5')
    return 0


if __name__ == "__main__":
    import sys
    for place in sys.path:
        print(place)

    # データを読み込むフォルダを指定する
    DirPath = './data/CNN_test'
    img_width, img_height = 64, 64
    CLS = np.arange(0, 180, 5).astype('str')

    train(DirPath, img_size=(img_width, img_height), cls_label=CLS)
