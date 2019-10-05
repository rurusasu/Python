# coding: utf-8

from functions import _CallFunction
import numpy as np


class Sequential:
    #----------------------
    # 汎用関数
    #----------------------




    def __init__(self):
        self.sequential = {}
        self.units = {}
        self.func = {}
        self.frontUnit = 0
        self.loss = None
        self.optimizer = None
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
        """
        if (self.i != 1):
            # 入力行列 = (前層のユニット数, 現在の層のユニット数)
            self.units[self.i-1] = (self.frontUnit, self.sequential[self.i].units)
        # 前層のユニット数を更新
        self.frontUnit = self.sequential[self.i].units
        """
        self.units[self.i] = self.sequential[self.i].units
        self.i += 1


    def compile(self, loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']):
        self.func['loss'] = loss
        self.func['optimizer'] = optimizer
        self.func['metrics'] = metrics

        for i in range(len(self.units)):
            if (i == len(self.units)-1):
                Tuple = (self.units[i+1], 1)
            else:
                Tuple = (self.units[i+1], self.units[i+2])
            self.sequential[i+1]._initParams(Tuple[1])

        # それぞれのunit内部関数を設定（例：affine + relu）
        self.sequential[i].setFunc(lr=0.01)
        

    def _loss(self, loss):
        return _CallFunction('functions', loss)


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
            TrainI_batch, TrainT_batch = self._classif(training_input, training_test, batch_size)

            for i in range(batch_size):
                self.gradient(TrainI_batch, TrainT_batch)
        
            
        print(self.units)
        print(self.sequential)

    def gradient(self, input, test):
        test = test
        y = self._predict(input)
        #print(y.shape)


    def _classif(self, trainI, trainT, batch_size):
        Input_rowSize = trainI.shape[0]
        # 行数からbatch_sizeだけランダムに値を抽出 replace(重複)
        if (batch_size >= Input_rowSize):
            batch_mask = np.random.choice(Input_rowSize, batch_size)
        elif (batch_size < Input_rowSize):
            batch_mask = np.random.choice(
                        Input_rowSize, batch_size, replace=False)
        # 全データからbatch_size分データを抽出
        TrainI_batch = trainI[batch_mask, 0:]
        TrainT_batch = trainT[batch_mask]

        return TrainI_batch, TrainT_batch

    
    def _predict(self, x):
        for layer in self.sequential.values():
            x = layer.forward(x)
        return x
    



if __name__ == "__main__":
    from layers_2 import*
    training_input = np.array([np.arange(0, 5, 0.1), np.arange(10, 15, 0.1)])
    training_test  = np.arange(90, 100, 0.2)

    model = Sequential()
    model.add(InputLayer(input_shape=(3,)))
    model.add(Dense(50, weight_initializer='sigmoid'))
    model.add(Dense(50, weight_initializer='he'))
    model.add(Dense(3, activation='linear'))

    epoch = 20
    batch = 50

    model.compile()
    model.fit(training_input, training_test, batch, epoch)

