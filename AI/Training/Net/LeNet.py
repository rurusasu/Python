import keras
import cv2
import numpy as np
import argparse
from DataAugumentation import DataLoad as Load
from glob import glob

#import DataAugumentation

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


def tarin(DirPath, epoches):
    """
    LeNetの訓練を行う関数

    Parameters
    ----------
    DirPath: str
        訓練画像ディレクトリへのパス
    epoches: int
        学習回数
    """
    model = LeNet()

    for layer in model.layers:
        layer.trainable =True

    model.compile(
        loss = 'categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy']
    )

    xs, ts, paths = Load.Data_load(DirPath, Framework='Keras')

    for i in range(epoches):
        # training
        x, t = Load.MiniBatch(xs, ts, mb=8, mbi=0)
        loss, acc = model.train_on_batch(x=x, y=t)
        print('iter >>', i+1, ', loss >>', loss, ', accuracy >>', acc)

    model.save('model.h5')


# test
def test(DirPath):
    # load training model
    model = LeNet()
    model.load_weights('model.h5')

    xs, ts, paths = Load.Data_load(DirPath, Framework='Keras')

    for i in range(len(paths)):
        x = xs[i]
        t = ts[i]
        path = paths[i]

        x = np.expand_dims(x, axis=0)

        pred = model.predict_on_batch(x)[0]
        print('in {}, predicted probabilities >> {}', format(path, pred))

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    """
    args = arg_parse()

    if args.train:
        tarin('../../DataSet/AngleDetection/training/*', 500)
    if args.test:
        test('../../DataSet/AngleDetection/test/*')
    
    if not (args.train or args.test):
        print('please select train or test flag')
        print('train: python main.py --train')
        print('test:  python main.py --test')
        print('both:  python main.py --train --test')
    """