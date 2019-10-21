import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.functions import _CallClass, _CallFunction
from common.layers import *
from collections import OrderedDict


class Sequential:
    def __init__(self):
        self.sequential = []
        self.func = {}
        self.func['loss'] = None
        self.func['optimizer'] = None

        self.history = {}
        self.history['loss'] = []
        self.history['loss_ave'] = []

    def add(self, layer_name):
        #リストにレイヤの名前を代入
        self.sequential.append(layer_name)

    def compile(self, loss, optimizer='sgd'):
        self.func['loss'] = _CallClass('common.layers', loss)
        self.func['loss'] = self.func['loss']()
        self.func['optimizer'] = _CallClass('common.optimizer', optimizer)
        self.func['optimizer'] = self.func['optimizer']()
        #self.LastLayer = globals()[loss]()

   #-------------------------------------------------
   # fit:学習のメイン
   #     引数
   #     @self
   #     @training_input:学習データ（入力）
   #     @training_test :教師データ
   #     @epochs        :エポック数
   #     @epsilon=0.01  :学習率（初期値0.01）
   #     変数
   #     @TrainingI       :TrainingInputの略
   #     @TrainigT        :TrainigTestの略
   #     @IRS             :InputRowSizeの略(training_inputの行数 返り値：整数)
   #     @ICS             :InputColSizeの略(training_inputの列数 返り値：整数)
   #     @batch_size      :バッチ数
   #     @TrainingI_batch :TrainingIからバッチ数個だけデータを抽出した行列
   #     @TrainingT_batch :TrainingTからバッチ数個だけデータを抽出した行列
   #-------------------------------------------------

    def fit(self, x, t, batch_size, epochs):
        #plot = Plot(0, 1)

        IRS = x.shape[0]
        ICS = t.shape[1]

        #ValidationData
        #x_val_data = validation_data[0] # ValidationDataの行数を取得 返り値：整数
        #t_val_data = validation_data[1] # ValidationDataの列数を取得 返り値：整数

        #ValidationRow_size = x_val_data.shape[0]
        #ValidationCol_size = x_val_data.shape[1]

        #レイヤの行列を計算する
        y = np.zeros((3, batch_size))  # (128, 3) , (3, 128)
        for layers in self.sequential:
            y = layers.fit(y)

        # メインルーチン
        for i in range(epochs):
            # 行数からbatch_sizeだけランダムに値を抽出 replace(重複)
            batch_mask = np.random.choice(IRS, batch_size, replace=False)
            x_batch = x[batch_mask, 0:ICS]  # 全データからbatch_size分データを抽出
            t_batch = t[batch_mask]

            #x_val   = x_val_data[batch_mask, 0:ValidationCol_size]
            #t_val   = t_val_data[batch_mask]

            out_ave = 0
            loss_ave = 0

            y = self.Predict(x_batch)  # 学習を行う
            loss = self.func['loss'].forward(y, t_batch)
            loss_ave = loss / batch_size
            self.history['loss_ave'].append(loss_ave)

            ########################
            #####     追加     #####
            ########################
            #逆伝播を行うためにレイヤを反転
            self.sequential.reverse()

            #逆伝搬および重みの更新
            dout = self.func['loss'].backward(out_ave, loss_ave)
            for layer in self.sequential:
                dout = layer.backward(dout)

            self.sequential.reverse()
            print('学習%d回目  --loss:%f' % (i+1, loss_ave))

        print('loss = %f' % self.history['loss_ave'][epochs-1])
        return self.history

    ########################################
    ########      内部関数       ###########
    ########################################

    def Predict(self, x):
        for layers in self.sequential:
            x = layers.forward(x)

        return x
