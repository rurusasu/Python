# coding: utf-8

from functions import _CallFunction
import numpy as np


class Sequential:
    counter = 1

    def __init__(self):
        self.sequential = {}
        self.units = {}
        self.counter = 1
        """
        self.Output = output()
        self.Output.history = {}
        self.OutputBuff = []
        self.Output.history['loss'] = []
        self.Output.history['loss_ave'] = []
        """

    def add(self, layer_name):
        #リストにレイヤの名前を代入
        self.sequential[self.counter] = layer_name
        self.units[self.counter] = self.sequential[self.counter].units
        self.counter += 1
        #print(self.sequential)
        print(self.units)


    def compile(self, loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']):
        for i in range(len(self.units)):
            print(i+1)
            print(self.sequential[i+1])
            self.sequential[i+1]._initParams
        #self.LastLayer = globals()[loss]()


    def _loss(self, loss):
        methoed = _CallFunction('functions', loss)
        loss = method()


    def _optimizer(self, optimizer):
        method = _CallFunction('optimizer', optimizer)
    

    def fit(self, training_input, training_test, batch_size, epochs, validation_data, epsilon=0.01, reg_lambda=0.01):
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

        plot = Plot(0, 1)

        IRS = training_input.shape[0]
        ICS = training_input.shape[1]

        #レイヤの行列を計算する
        y = ICS
        for layers in self.sequential:
            y = layers.unit(y, Sequential.counter, epsilon, reg_lambda)
            Sequential.counter += 1

        Sequential.counter = 1
        # メインルーチン
        for i in range(epochs):
            # 行数からbatch_sizeだけランダムに値を抽出 replace(重複)
            batch_mask = np.random.choice(IRS, batch_size, replace=False)
            # 全データからbatch_size分データを抽出
            TrainingI_batch = training_input[batch_mask, 0:ICS]
            TrainingT_batch = training_test[batch_mask]

            #x_val   = x_val_data[batch_mask, 0:ValidationCol_size]
            #t_val   = t_val_data[batch_mask]

            print('#######    学習%d回目    ########' % Sequential.counter)
            Sequential.counter += 1

            out_sum = 0
            out_ave = 0
            loss_sum = 0
            loss_ave = 0
            for j in range(batch_size):
                output = self.Predict(TrainingI_batch[j, :])  # 学習を行う

                out_sum += output
                #####     誤差を保存する     #####
                loss = 0
                loss = self.LastLayer.forward(output, TrainingT_batch[j])
                loss_sum += loss
                self.Output.history['loss'].append(loss)

            out_ave = out_sum / batch_size
            loss_ave = loss_sum / batch_size
            plot.grah_plot(i+1, loss_ave)
            self.Output.history['loss_ave'].append(loss_ave)

            ########################
            #####     追加     #####
            ########################
            #逆伝播を行うためにレイヤを反転
            self.sequential.reverse()

            #逆伝搬および重みの更新
            dout = self.LastLayer.backward(out_ave, loss_ave)
            for layer in self.sequential:
                dout = layer.backward(dout)

            self.sequential.reverse()

        print('loss = %f' % self.Output.history['loss'][epochs-1])
        return self.Output


if __name__ == "__main__":
    from layers_2 import*
    model = Sequential()
    model.add(Dense(50))
    model.add(Dense(50))

    model.compile()