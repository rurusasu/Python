import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers

import numpy as np
import DataLoad_0602 as Load


activation = layers.Activation(
    'sigmoid',  # 活性化関数(隠れ層用): sigmoid関数(変更可能)
    name='activation'  # 活性化関数にも名前付け
)

acti_out = layers.Activation(
    'softmax',  # 活性化関数(出力層用): softmax(固定)
    name='acti_out'  # 活性化関数似も名前付け
)


class LeNet(Model):
    ##################
    #  レイヤーを定義  #
    ##################
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #assert len(opt) != 0
        img_height, img_width = opt['img_height'], opt['img_width']
        num_classes = opt['num_classes']

        # 隠れ層：1つ目のレイヤー
        self.layer1 = layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            padding='valid',
            activation=None,
            name='conv1'
        )
        # 隠れ層：2つ目のレイヤー
        self.layer2 = layers.MaxPool2D(
            pool_size=(2, 2),
            padding='same',
            name='MaxPool1'
        )

        # 隠れ層：4つ目のレイヤー
        self.layer4 = layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            padding='valid',
            activation=None,
            name='conv2'
        )

        # 隠れ層：5つ目のレイヤー
        self.layer5 = layers.MaxPool2D((2, 2), padding='same')

        self.layer7 = layers.Flatten()
        self.layer8 = layers.Dense(120, name='dense1', activation=None)
        self.layer9 = layers.Dense(64, name='dense2', activation=None)
        self.layer_out = layers.Dense(
            num_classes, name='output', activation='softmax')

        #model = Model(inputs=inputs, outputs=x, name='model')
        # return model

    def call(self, inputs, training=None):
        # [出力=活性化関数(第n層(入力))]の形式で記述
        x = self.layer1(inputs)
        x = activation(self.layer2(x))
        x = activation(self.layer5(x))
        outputs = acti_out(self.layer_out(x))
        return outputs


def train(DirPath, opt):
    model = LeNet()  # モデルの生成

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    xs, ts, paths, img_read_err, err_path = data_load(
        DirPath, opt['img_width'], opt['img_height'], opt['num_classes'])
    if img_read_err:
        print('画像の読み込みでエラーが発生しました．')
        print(img_read_err)
        pass
    elif err_path:
        print('one_hot_lbl作成時にエラーが発生しました．')
        print(err_path)
        pass

    # training
    mb = 100
    mbi = 0
    count = 1
    loss_ls = []

    loss_ave = 0
    loss_AveMin = 1
    loss_AveCnt = 0
    #loss_AveLs = []

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

            loss_ave = np.average(loss_ls)
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
        #print('iter >>', count, ',loss >>', loss, 'accuracy >>', acc)

        loss_ls.append(loss)
        count += 1

        if loss_AveCnt == 50:
            break

    model.save('LeNet.h5')
    return 0


if __name__ == "__main__":

    opt = {
        'img_height': 64,
        'img_width': 64,
        'num_classes': 36,
    }
    # データを読み込むフォルダを指定する
    DirPath = './data/CNN_test'
    img_width, img_height = 64, 64
    CLS = np.arange(0, 180, 5).astype('str')

    load = Load.DataLoad(DirPath, 'jpg', size=(img_width, img_height))
    xs, ts = load.data_load()

    model1 = LeNet(opt, name='subclassing_model1')  # モデルの生成
    # 入力方法を指定して計算グラフを構築する7
    model1.build(
        input_shape=(None, img_width, img_height, 3)
    )

    #temp_input = xs[1]
    #temp_output = model1.predict(temp_input)

    # モデルの内容を出力
    model1.summary()
    #train(DirPath, img_size=(64, 64), cls_label=CLS)
    #train(DirPath, opt)
