# coding: utf-8

import sys, os
sys.path.append(os.getcwd())
from collections import OrderedDict
from common.functions import _CallFunction, _CallClass
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

        #self.Output = output()
        self.history = {}
        #self.OutputBuff = []
        self.history['loss'] = []
        self.history['loss_ave'] = []
        

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
        self.func['loss'] = _CallFunction('functions', loss)
        self.func['optimizer'] = _CallClass('optimizer', optimizer)
        #self.func['metrics'] = metrics



    def fit(self, input, test, batch_size, epochs, lr=0.01, reg_lambda=0.01):
        """
        fit
    
        Parameters
        ----------
        input : 
            学習データ（入力）
        test :
            教師データ
        epochs :
            エポック数
        epsilon=0.01 :
            学習率（初期値0.01）
        """
        """
        for i in range(len(self.units)):
            if (i == len(self.units)-1):
                Tuple = (self.units[i+1], 1)
            else:
                Tuple = (self.units[i+1], self.units[i+2])
            self.sequential[i+1].fit(Tuple[1])
        """

        # レイヤfit
        # 入力データと同じ配列のゼロ行列を作成
        # 1.InputLayerにデータを通して、成型
        # 2.InputLayerから、データの行数および列数をタプルとしてreturn
        # 3.Denseにタプルを通し、行列数を決定
        # 4.Denseの出力を出力層に入力し、層の行列を決定
        zeros = np.zeros((batch_size, input.shape[1]))
        y = self.sequential[1].fit(zeros)
        for i in self.sequential.keys():
            print(i)


        # メインルーチン
        for n in range(epoch):
            input_bat, test_bat = self.__classif__(input, test, batch_size)

            # データの成型
            input_bat, test_bat = self.__Molding__(input_bat, test_bat)

            """
            for i in range(batch_size):
                mask = self.__fitMask__(TrainI_batch, self.units[1])
                self.gradient(TrainI_batch[mask, 0:], TrainT_batch[mask])
            """
            

            loss = self.__loss__(input_bat, test_bat)
            #loss = self.__loss__(input_bat[mask, 0:], test_bat[mask])
            self.history['loss'].append(loss)

        print(self.history['loss'])


    def __Molding__(self, input, test):
        # 入力層の行数と入力データの行数が等しいとき
        # もしくは入力層の行数と入力データの列数が等しいとき
        if (input.shape[0] == self.units[1] or 
                    input.shape[1] == self.units[1]):
            input = input.T
            test = test.T
            return input, test
        
        # どちらとも等しくないとき
        else:
            print('Data Input Error')
            return None, None


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
        # 入力データの行数を保存
        Input_rowSize = trainI.shape[0]
        # 行数からbatch_sizeだけランダムに値を抽出 replace(重複)
        if (batch_size > Input_rowSize):
            print('batch_choice エラー！')
            return None
        elif (batch_size <= Input_rowSize):
            batch_mask = np.random.choice(
                Input_rowSize, batch_size, replace=False)
        # 全データからbatch_size分、行を抽出
        TrainI_batch = trainI[batch_mask]
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
        return self.func['loss'](y, t)

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
    model.add(InputLayer(input_shape=(50,)))
    model.add(Dense(50, weight_initializer='sigmoid'))
    model.add(Dense(50, weight_initializer='he'))
    #model.add(Dense(3, activation='linear'))
    #model.add(Dense(3, weight_initializer='he'))

    epoch = 5
    batch = 3

    model.compile()
    model.fit(training_input, training_test, batch, epoch)
    #model._GetLayerParams()
