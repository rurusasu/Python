import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import math
from nn_pt3.functions import *
from nn_pt3.layers import *
from collections import OrderedDict

class output: pass


class Plot:
    def __init__(self, x, y):
        #グラフの初期化
        self.fig, self.ax = plt.subplots(1, 1)
        self.x = []
        self.y = []

        self.x.append(x)
        self.y.append(y)

        self.lines,  = self.ax.plot(self.x, self.y)

    def grah_plot(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.lines.set_data(self.x, self.y)
        self.ax.set_xlim(0, len(self.x))
        self.ax.set_ylim(min(self.y), max(self.y))
        plt.pause(.01)





class Sequential:
    counter  = 1

    def __init__(self):
        self.sequential = []
        self.Output = output()
        self.Output.history = {}

        self.OutputBuff = []
        self.Output.history['MiniBatchLoss'] = []
        self.Output.history['BatchLoss']     = []
        #self.Output.history['val_loss'] = []
        #self.Output.history['acc']      = []
        #self.Output.history['val_acc']  = []

    def add(self, layer_name):
        #リストにレイヤの名前を代入
        self.sequential.append(layer_name)

    def compile(self, loss):
        #self.LastLayer = globals()[loss]()
        self.loss = Loss(loss)

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
   #     @IRS_Buff        :InputRowSizeの略(training_inputの行数 返り値：整数)
   #     @ICS_Buff        :InputColSizeの略(training_inputの列数 返り値：整数)
   #     @batch_size      :バッチ数
   #     @TrainingI_batch :TrainingIからバッチ数個だけデータを抽出した行列
   #     @TrainingT_batch :TrainingTからバッチ数個だけデータを抽出した行列
   #-------------------------------------------------
    def fit(self, training_input, training_test, batch_size, epochs, validation_data, epsilon=0.01, reg_lambda=0.01):
        TrainingI_Buff = training_input
        TrainingT_Buff = training_test

        IRS_Buff = training_input.shape[0]
        ICS_Buff = training_input.shape[1]
      
        BatchSize_Buff = batch_size

        MiniBatchSize = 256
        
        Dchoice = DataChoice()
        plot = Plot(0, 1)

        # もし、BatchSizeが1000より小さいとき
        if (BatchSize_Buff < 1000):

            # もし、全データ数がBatchSizeで割り切れる場合
            if (IRS_Buff % BatchSize_Buff == 0):
                iteration = IRS / BatchSize # iterationは全データ数÷BatchSizeにする

            # もし、全データ数がBatchSizeで割り切れない場合
            elif (IRS_Buff % BatchSize_Buff != 0):
                iteration = int(IRS_Buff / BatchSize_Buff) + 1 # (全データ数÷BatchSize)の小数点を切り捨てた数+1にする
                RestBatchSize = IRS_Buff % BatchSize_Buff # Batchの最後に余るデータ数
                #LastBatchSize = RestBatchSize + (BatchSize - RestBatchSize) # 最後のBatch数を拡張した数
                               
            # それ以外の場合
            else:
                print('iterationが決定できませんでした。')
                return

                       
            #レイヤの行列を計算する
            y = ICS_Buff
            for layers in self.sequential:
                y = layers.unit(BatchSize_Buff, y, Sequential.counter, epsilon, reg_lambda)
                Sequential.counter += 1

            Sequential.counter = 1
            # メインルーチン
            for i in range(epochs):
                print('#######    学習%d回目    ########' %Sequential.counter)
                Sequential.counter += 1

                TrainingI_batch = TrainingI_Buff
                TrainingT_batch = TrainingT_Buff

                IRS = IRS_Buff
                #BatchSize = BatchSize_Buff
                for j in range(iteration):
                    if IRS % BatchSize_Buff != 0:
                        I_batch = Dchoice.RandomChoice(TrainingI_batch, IRS, BatchSize_Buff)
                        T_batch = Dchoice.RandomChoice(TrainingT_batch, IRS, BatchSize_Buff)

                        ##### 使用したデータを削除 #####
                        TrainingI_batch = Dchoice.DataDelete(TrainingI_batch, delete_size = RestBatchSize)
                        TrainingT_batch = Dchoice.DataDelete(TrainingT_batch, delete_size = RestBatchSize)

                        IRS -= RestBatchSize

                    else:
                        I_batch = Dchoice.RandomChoice(TrainingI_batch, IRS, BatchSize_Buff)
                        T_batch = Dchoice.RandomChoice(TrainingT_batch, IRS, BatchSize_Buff)

                        ##### 使用したデータを削除 #####
                        TrainingI_batch = Dchoice.DataDelete(TrainingI_batch)
                        TrainingT_batch = Dchoice.DataDelete(TrainingT_batch)

                        IRS -= BatchSize_Buff

                    # 学習を行う
                    output = self.Predict(I_batch)

                    ##### 誤差を計算する #####
                    loss = self.loss.forward(output, T_batch, BatchSize_Buff, sum=1)
                    self.Output.history['MiniBatchLoss'].append(loss)


                BatchLoss, BackSignal = self.loss.backward(self.Output.history['MiniBatchLoss'], iteration)
                plot.grah_plot(i+1, BatchLoss)
                self.Output.history['BatchLoss'].append(BatchLoss)
                self.Output.history['MiniBatchLoss'] = [] # 配列の初期化
            
                #逆伝播を行うためにレイヤを反転
                self.sequential.reverse()

                #逆伝搬および重みの更新
                for layer in self.sequential:
                    BackSignal = layer.backward(BackSignal)
            
                self.sequential.reverse()
        
            print('loss = %f' %self.Output.history['loss'][epochs-1])
            return self.Output

        '''
        # もし、BatchSizeが1000以上 かつ BatchSizeがMiniBatchSize以上    かつ  BatchSizeがMiniBatchSizeで割り切れるとき
        if (BatchSize >= 1000) and (BatchSize >= MiniBatchSize) and (BatchSize % MiniBatchSize == 0):
            cycle = BatchSize / MiniBatchSize

        # もし、BatchSizeが1000以上   かつ BatchSizeがMiniBatchSize以上    かつ  BatchSizeがMiniBatchSizeで割り切れない
        elif (BatchSize >= 1000) and (BatchSize >= MiniBatchSize) and (BatchSize % MiniBatchSize != 0):
            cycle = int(BatchSize / MiniBatchSize) + 1 # (BatchSize÷MiniBatchSize)の小数点を切り捨てた数+1にする
            RestBatchSize = BatchSize % MiniBatchSize
            LastBatchSize = RestBatchSize + (MiniBatchSize - RestBatchSize)

        # もし、BatchSizeが1000以上   かつ BatchSizeがMiniBatchSizeより小さいとき
        elif (BatchSize >= 1000) and (BatchSize < MiniBatchSize):
            MiniBatchSize = 256

        # BatchSizeが1000より小さいとき
        else:
            MiniBatchSize = BatchSize_Buff

        Dchoice = DataChoice()

        plot = Plot(0, 1)
        

        #レイヤの行列を計算する
        y = ICS_Buff
        for layers in self.sequential:
            y = layers.unit(MiniBatchSize, y, Sequential.counter, epsilon, reg_lambda)
            Sequential.counter += 1

        Sequential.counter = 1
        # メインルーチン
        for i in range(epochs):
            print('#######    学習%d回目    ########' %Sequential.counter)
            Sequential.counter += 1

            TrainingI_batch = TrainingI_Buff
            TrainingT_batch = TrainingT_Buff

            IRS = IRS_Buff
            for j in range(iteration):

                if batch_size <= 1000:
                    I_batch = Dchoice.RandomChoice(TrainingI_batch, IRS, BatchSize)
                    T_batch = Dchoice.RandomChoice(TrainingT_batch, IRS, BatchSize)
                    #batch_mask = np.random.choice(IRS, BatchSize, replace = False) 
                    #TrainingI_batch = training_input[batch_mask, 0:ICS]
                    #TrainingT_batch = training_test[batch_mask]

                    output = self.Predict(I_batch) # 学習を行う

                    ##### 誤差を計算する #####
                    loss = self.loss.forward(output, T_batch, MiniBatchSize, sum=1)
                    self.Output.history['MiniBatchLoss'].append(loss)

                    ##### 使用したデータを削除 #####
                    TrainingI_batch = Dchoice.DataDelete(TrainingI_batch)
                    TrainingT_batch = Dchoice.DataDelete(TrainingT_batch)
                    #TrainingI_batch = np.delete(TrainingI_batch, BatchMask, 0)
                    #TrainingT_batch = np.delete(TrainingT_batch, BatchMask)
                    IRS -= BatchSize

                # batch_sizeが1000よりも大きい場合
                # MiniBatchを利用して計算
                else:
                    # batch_sizeがminibatch_sizeで割り切れない場合
                    if BatchSize % MiniBatchSize != 0:
                        I_LastBatch = Dchoice.RandomChoice(TrainingI_batch, BatchSize, LastBatchSize)
                        T_LastBatch = Dchoice.RandomChoice(TrainingT_batch, BatchSize, LastBatchSize)
                        #LastBatchMask = np.random.choice(BatchSize, LastBatchSize, replace = False)
                        #TrainingI_LastBatch = TrainingI_batch[LastBatchMask, 0:ICS]
                        #TrainingT_LastBatch = TrainingT_batch[LastBatchMask]
                        TrainingI_batch = Dchoice.DataDelete(TrainingI_batch)
                        TrainingT_batch = Dchoice.DataDelete(TrainingT_batch)
                        #TrainingI_batch = np.delete(TrainingI_batch, LastBatchMask, 0)
                        #TrainingT_batch = np.delete(TrainingT_batch, LastBatchMask)
                        BatchSize = BatchSize - RestBatchSize

                    # 0からBatchSize_BuffまでMiniBatchSizeごとに変化させる
                    for k in range(0, BatchSize_Buff, MiniBatchSize):
                        TrainingI_MiniBatch = DataChoice.RandomChoice(TrainingI_batch, BatchSize, MiniBatchSize)
                        TrainingT_MiniBatch = DataChoice.RandomChoice(TrainingT_batch, BatchSize, MiniBatchSize)
                        #MiniBatchMask = np.random.choice(BatchSize, MiniBatchSize, replace = False)
                        #TrainingI_MiniBatch = TrainingI_batch[MiniBatchMask, 0:ICS]
                        #TrainingT_MiniBatch = TrainingT_batch[MiniBatchMask]
                        output = self.Predict(TrainingI_MiniBatch) # 学習を行う

                        ##### 誤差を計算する #####
                        loss = self.loss.forward(output, TrainingT_MiniBatch, MiniBatchSize, sum=1)
                        self.Output.history['MiniBatchLoss'].append(loss)
                        
                        ##### 使用したデータを削除 #####
                        TrainingI_MiniBatch = DataChoice.DataDelete(TrainingI_MiniBatch)
                        TrainingT_MiniBatch = DataChoice.DataDelete(TrainingT_MiniBatch)
                        #TrainingI_batch = np.delete(TrainingI_batch, MiniBatchMask, 0)
                        #TrainingT_batch = np.delete(TrainingT_batch, MiniBatchMask)
                        BatchSize = BatchSize - MiniBatchSize

            BatchLoss, BackSignal = self.loss.backward(self.Output.history['MiniBatchLoss'], BatchSize_Buff)
            plot.grah_plot(i+1, BatchLoss)
            self.Output.history['BatchLoss'].append(BatchLoss)
            self.Output.history['MiniBatchLoss'] = [] # 配列の初期化
            
            #逆伝播を行うためにレイヤを反転
            self.sequential.reverse()

            #逆伝搬および重みの更新
            for layer in self.sequential:
                BackSignal = layer.backward(BackSignal)
            
            self.sequential.reverse()
        
        print('loss = %f' %self.Output.history['loss'][epochs-1])
        return self.Output
    '''

    ########################################
    ########      内部関数       ###########
    ########################################
    def Predict(self, data_set):
        for layers in self.sequential:
            data_set = layers.forward(data_set)

        return data_set


    def ValLoss(self, val_data, t_val):
        val_data = self.Predict(val_data)
        val_loss = self.LastLayer.forward(val_data, t_val)

        return val_loss


    def accuracy(self, x, t):
        y = self.Predict(x)
        #y = np.argmax(y, axis = 1)
        #if t.ndim != 1 : t = np.argmax(t, axis = 1)

        if y.shape != t.shape:
            y = y.reshape(-1, 1)
            t = t.reshape(-1, 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy



#訓練データの読み込み
data = np.loadtxt(
    "save_data.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )

#テストデータの読み込み
test = np.loadtxt(
    "test_data.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )



#読み込んだデータを学習用にコピーする
data_1 = data_copy(data)
test_1 = data_copy(test)

#標準化
data_ = data_std(data_1)
test_ = data_std(test_1)

#正規化
data_ = data_nom(data_)
test_ = data_nom(test_)

#訓練データのセット
training_input = data_[:, 0:2] #学習データをセット
training_test = data_[:, 2]   #教師データをセット

#テストデータのセット
x_test  = test_[:, 0:2] #入力データをセット
t_test  = test_[:, 2]   #正解データをセット


module = Sequential()
module.add(InputLayer(input_shape = (2, )))
#module.add(Dense(50, activation = 'sigmoid'))
#module.add(Dense(50, activation = 'sigmoid'))
module.add(Dense(50, activation = 'relu'))
module.add(Dense(50, activation = 'relu'))
module.add(Dense(1,  activation = 'liner'))
module.compile(loss = 'MeanSquaredError')

#学習
epochs = 20
batch_size = 32

# Gradient descent parameters (数値は一般的に使われる値を採用) 
epsilon = 0.01    # gradient descentの学習率
reg_lambda = 0.01 # regularizationの強さ 

history = module.fit(training_input, training_test, batch_size=batch_size, epochs=epochs, validation_data = (x_test, t_test), epsilon=epsilon, reg_lambda=reg_lambda)

