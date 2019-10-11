# coding: utf-8

import sys, os
sys.path.append(os.getcwd())
from collections import OrderedDict
from common.functions import _CallFunction
from common.gradient import numerical_gradient_2d
import numpy as np

class Sequential:
    #----------------------
    # 汎用関数
    #----------------------
    def __init__(self):
        self.sequential = OrderedDict()
        self.units = {}
        self.func = {}
        self.func['loss'] = None
        self.func['optimizer'] = None
        self.func['metrics'] = None
        self.i = 1
        """
        self.Output = output()
        self.Output.history = {}
        self.OutputBuff = []
        self.Output.history['loss'] = []
        self.Output.history['loss_ave'] = []
        """

    def add(self, layer_name):
        #リストにレイヤの名前を代入
        self.sequential[self.i] = layer_name
        self.units[self.i] = self.sequential[self.i].units
        self.i += 1


    def _GetLayerParams(self):
        # sequentialに格納されているオブジェクトを表示
        print(self.sequential.values())
        for i in range(len(self.units)):
            self.sequential[i+1]._GetParams()


    def compile(self, loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']):
        self.func['loss'] = loss
        self.func['optimizer'] = optimizer
        self.func['metrics'] = metrics

        for i in range(len(self.units)):
            if (i == len(self.units)-1):
                Tuple = (self.units[i+1], 1)
            else:
                Tuple = (self.units[i+1], self.units[i+2])
            self.sequential[i+1].compile(Tuple[1])


    def fit(self, training_input, training_test, batch_size, epochs, lr=0.01, reg_lambda=0.01):
        """
        fit
    
        Parameters
        ----------
        self
        training_input : 
            学習データ（入力）
        training_test :
            教師データ
        epochs :
            エポック数
        epsilon=0.01 :
            学習率（初期値0.01）
        """

        # メインルーチン
        for n in range(epoch):
            TrainI_batch, TrainT_batch = self.__classif__(training_input, training_test, batch_size)

            for i in range(batch_size):
                mask = self.__fitMask__(TrainI_batch, self.units[1])
                """
                if (-1 == mask):
                    print('Input Error')
                    return
                else:
                    self.gradient(TrainI_batch[mask, 0:], TrainT_batch[mask])
                """
                self.gradient(TrainI_batch[mask, 0:], TrainT_batch[mask])
    
    def __classif__(self, trainI, trainT, batch_size):
        """
        すべてのデータからバッチ数分だけデータを抽出する関数

        Parameters
        ----------
        trainI : numpy.ndarray
            学習用の入力データ
        trainT : numpy.ndarray
            学習用のテストデータ
        batch_size : int
            バッチサイズ

        Returns
        -------
        TrainI_batch : numpy.ndarray
            バッチ数だけ抽出した学習用入力データ
        TrainT_batch : numpy.ndarray
            バッチ数だけ抽出した学習用テストデータ
        """
        Input_rowSize = trainI.shape[0]
        # 行数からbatch_sizeだけランダムに値を抽出 replace(重複)
        if (batch_size > Input_rowSize):
            print('batch_choice エラー！')
            return None
        elif (batch_size <= Input_rowSize):
            batch_mask = np.random.choice(
                Input_rowSize, batch_size, replace=False)
        # 全データからbatch_size分データを抽出
        TrainI_batch = trainI[batch_mask, 0:]
        TrainT_batch = trainT[batch_mask]

        return TrainI_batch, TrainT_batch

    def __fitMask__(self, data, node_num):
        """
        入力層のノード数と入力データの行数を比較する

        Parameter
        ---------
        data : numpy.ndarray
            入力データ

        Return
        ------
        data : numpy.ndarray
            出力データ
        """
        data_rowSize = data.shape[0]
        # 入力データの行数と入力層のノード数が
        # 入力データの行数が多い場合
        if (data_rowSize >= node_num):
            mask = np.random.choice(data_rowSize, node_num)
            return mask
        # ノード数が多い場合
        else:
            print('batch_size < InputLayer')


    def gradient(self, input, test):
        # forward
        dout = self.__loss__(input, test)
        print(dout.shape)
        # backward
        revSequence = list(self.sequential.values())
        revSequence.reverse()
        for revLayer in revSequence:
            dout = revLayer.backward(dout)

    def __loss__(self, x, t):
        """
        誤差の計算を行う関数

        Parameters
        ----------
        x : numpy.ndarray
            入力データ
        t : numpy.ndarray
            教師データ

        Return
        ------
        loss : list
            誤差の計算結果
        """
        y = self.__predict__(x)
        loss = _CallFunction('functions', self.func['loss'])
        return loss(y, t)

    def __predict__(self, x):
        for layer in self.sequential.values():
            x = layer.forward(x)
        return x

    #def __optimizer__(self):
        #for layer in 


if __name__ == "__main__":
    from layers import*
    training_input = np.array(
        [np.arange(0, 5, 0.1), np.arange(10, 15, 0.1), np.arange(20, 25, 0.1)])
    training_test  = np.arange(90, 100, 0.2)

    model = Sequential()
    model.add(InputLayer(input_shape=(2,)))
    model.add(Dense(50, weight_initializer='sigmoid'))
    model.add(Dense(50, weight_initializer='he'))
    #model.add(Dense(3, activation='linear'))
    model.add(Dense(3, weight_initializer='he'))

    epoch = 5
    batch = 3

    model.compile()
    model.fit(training_input, training_test, batch, epoch)
    #model._GetLayerParams()
