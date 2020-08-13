import sys, os
sys.path.append(os.getcwd())

import pickle
import numpy as np
import matplotlib.pyplot as plt
from common.functions import _CallClass, _CallFunction
from common.functions import __shuffle__, __sorting__
from common.layers import Input, Dense
from collections import OrderedDict



class Sequential:
    def __init__(self):
        self.sequential = OrderedDict()
        self.Dense = OrderedDict()
        self.grads = OrderedDict()
        #self.units = {}
        self.params = {}
        # sequrntial内で使う関数
        self.func = {}
        self.func['loss'] = None

        # 精度
        self.metrics_func = {}    # 精度計算に使う関数の辞書
        self.logs = OrderedDict() # 計算結果保存用の辞書

        # 誤差
        self.history = {}  # 計算結果保存用の辞書
        self.history['loss'] = 0
        self.history['loss_ave'] = []  
        self.history['val_loss'] = []

        self.score = {}
        self.i = 1
        self.callbacks = None


    def printParams(self):
        for num in range(len(self.sequential)):
            print('レイヤ：' + str(self.sequential[num+1]) +
                  'ユニット数：' + str(self.sequential[num+1].units))


    def add(self, layer_name):
        #リストにレイヤの名前を代入
        #self.sequential[self.i] = layer_name
        if layer_name.name in 'input':
            self.inputLayer = [layer_name, layer_name.units]
        elif layer_name.name in 'Dense_':
            self.sequential[layer_name.name + str(self.i)] = [layer_name, layer_name.units]
            self.i += 1
        else:
            print('実装されていないレイヤ名です。')
        #self.sequential[layer_name.name] = layer_name
        #self.sequential[self.i] = layer_name

        #self.units[self.i] = self.sequential[self.i].units
        #self.i += 1

    def compile(self, loss, optimizer='sgd', lr=0.01, metrics=['r2', 'rmse']):
        self.func['loss'] = _CallClass('common.layers', loss)
        self.func['loss'] = self.func['loss']()
        self.func['optimizer'] = _CallClass('common.optimizer', optimizer)
        self.func['optimizer'] = self.func['optimizer']()
        #----------------------------------------
        #レイヤ内の設定
        #----------------------------------------
        idx = 1
        y = np.zeros((1, self.inputLayer[1]))
        for key, layer in self.sequential.items():
            #y = layer.compile(y, optimizer, lr)
            #key, y, params = layer.compile(y)
            y = layer[0].compile(y)
            if 'Dense' in key:
                #self.Dense[key + str(idx)] = layer
                self.params['W' + str(idx)] = layer[0].params['W']
                self.params['b' + str(idx)] = layer[0].params['b']
                idx += 1
        #----------------------------------------
        # 精度検証用の関数を設定
        #----------------------------------------
        param = ['train_', 'val_']
        for metric in metrics:
            if str(metric).lower() in ('r2'):
                metric = 'r2_score'
            if str(metric).lower() in ('rmse'):
                metric = 'rmse_score'
            #self.metrics_func[metric] = _CallFunction('common.functions', metric)
            self.func[metric] = _CallFunction('common.functions', metric)
            #--------------------------
            # 結果保存用の配列を設定
            #--------------------------
            for i in param:
                name = str(i) + str(metric)
                self.logs[name] = []
            #-----------------------------
            # テスト結果保存用の配列を設定
            #-----------------------------
            self.score[metric] = []


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

    def fit(self, x, t, batch_size, epochs, validation=None, callbacks=None):
        # コールバック関数
        self.callbacks =callbacks
        """
        if callbacks != None:
            one_epoch_end = []
            for call in callbacks:
                if ('one_epoch_end' in dir(call)):
                    one_epoch_end.append(call.one_epoch_end)
        """ 
        #-------------------------------
        # Validation
        #-------------------------------
        if validation != None:    # バリデーションが最初からセットされているとき
            x_val = validation[0]
            t_val = validation[1]
        else:
            x = __shuffle__(x)
            t = __shuffle__(t)
            x_val, x = __sorting__(x, 2000)
            t_val, t = __sorting__(t, 2000)

        loop = int(x.shape[0] / batch_size)  # 繰り返し回数
        #---------------------------
        # メインルーチン
        #---------------------------
        for epoch in range(epochs):
            loss_sum = 0
            for j in range(loop):
                # 行数からbatch_sizeだけランダムに値を抽出 replace(重複)
                batch_mask = np.random.choice(
                    x.shape[0], batch_size, replace=False)
                x_batch = x[batch_mask]  # 全データからbatch_size分データを抽出
                t_batch = t[batch_mask]
                
                grads = self.gradient(x_batch, t_batch)
                self.func['optimizer'].update(self.params, grads)
                #loss = self.gradient(x_batch, t_batch) # 誤差の計算
                loss_sum += (self.history['loss'] / batch_size)

            loss_ave = loss_sum / loop                 # 誤差の平均値の計算
            self.history['loss_ave'].append(loss_ave)  # 誤差の保存
            
            #---------------------------
            # validation誤差の計算
            #---------------------------
            val_loss = self.loss(x_val, t_val)
            val_loss_ave = val_loss / x_val[0].shape[0]
            self.history['val_loss'].append(val_loss_ave)

            #---------------------------
            # 正解率の計算
            #---------------------------
            # printで値を表示するためのリスト
            prt_train_acc = []
            prt_val_acc = []
            logs={}
            for key, List in self.logs.items():
                #metric = [x for x in self.metrics_func.keys() if x in key][0]
                metric = [x for x in self.func.keys() if x in key][0]
                if 'train' in key:
                    #acc_train = self.accuracy(x_batch, t_batch, self.metrics_func[metric])
                    acc_train = self.accuracy(
                        x_batch, t_batch, self.func[metric])
                    List.append(acc_train)
                    prt_train_acc.append(acc_train)
                    logs[key] = acc_train
                if 'val' in key:
                    #acc_val = self.accuracy(x_val, t_val, self.metrics_func[metric])
                    acc_val = self.accuracy(
                        x_val, t_val, self.func[metric])
                    List.append(acc_val)
                    prt_val_acc.append(acc_val)
                    logs[key] = acc_val
                

            #---------------------------
            # one_epoch_endコールバック
            #---------------------------
            if self.callbacks != None:
                for call in callbacks:
                    call.one_epoch_end(logs)

            #print('学習%d回目  --loss:%f, --val=%f, --train_acc=%f' % (i+1, self.history['loss_ave'][i], self.history['val_loss'][i], self.metrics_log['train'][i]))
            print('学習%d回目  --loss:%f, --val=%f, --train_acc=%f, --val_acc=%f' % \
                    (epoch+1, self.history['loss_ave'][epoch], self.history['val_loss'][epoch], np.min(prt_train_acc), np.min(prt_val_acc)))

        
        nb_epoch = epochs
        plt.plot(range(nb_epoch), self.history['loss_ave'], marker='.', label='loss')
        plt.plot(range(nb_epoch), self.history['val_loss'], marker='.', label='val_loss')
        plt.legend(loc='best', fontsize=10)
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # 再描画する
        plt.pause(0.001)
        plt.show()
        print('loss=%f, val=%f' % (self.history['loss_ave'][epochs-1], self.history['val_loss'][epochs-1]))
        return self.history

    def evaluate(self, x, t):
        #self.score['loss'] = []

        logs={}
        for i in range(x.shape[0]):
            y = x[i, :].reshape((1, -1))
            y = self.predict(y)
            for key, values in self.func.items():
                if key in 'loss':
                    #self.score['loss'].append(self.func['loss'].forward(y, t))
                    logs[key] = self.func['loss'].forward(y, t)
                else:
                    #acc = np.sum(metric(y, t)) / y.shape[0]
                    acc = np.sum(values(y, t)) / y.shape[0]
                    #self.score[key].append(acc)
                    logs[key] = acc
                #---------------------------
                # one_epoch_endコールバック
                #---------------------------
                if self.callbacks != None:
                    for call in callbacks:
                        call.one_epoch_end(logs)
        
        return self.score


    def flow(self, x):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.predict(x)


    ########################################
    ########      内部関数       ###########
    ########################################
    def __modeling__(self, x, t, size):
        # 入力層の行数と入力データの行数が等しいとき
        # もしくは入力層の行数と入力データの列数が等しいとき
        if (x.shape[0] == size):
            x = x.T
            t = t.T
            return x, t
        elif(x.shape[1] == size):
            return x, t
        # どちらとも等しくないとき
        else:
            print('InpuLayer fit \
                    Data Input Error')
            return None, None


    #def __callback__(self, arg):


    def predict(self, x):
        #for layer in self.sequential.values():
        for layer in self.sequential.values():
            x = layer[0].forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.func['loss'].forward(y, t)

    def gradient(self, x, t):
        """
        関数の勾配を求める。
        """
        #forward
        self.history['loss'] = 0
        self.history['loss'] = self.loss(x, t)

        # backward
        #逆伝播を行うためにレイヤを反転
        #layers = list(self.sequential.values())
        layers = list(self.sequential.values())
        #layers = layers[0]
        layers.reverse()

        #逆伝搬および重みの更新
        dout = 1
        dout = self.func['loss'].backward()
        for layer in layers:
            dout = layer[0].backward(dout)
        del layers

        grads = {}
        idx = 1
        for layer in self.sequential.values():
            grads['W' + str(idx)] = layer[0].params['dW']
            grads['b' + str(idx)] = layer[0].params['db']
            idx += 1

        return grads

    def accuracy(self, x, t, metric):
        y = self.predict(x)

        acc = np.sum(metric(y, t)) / y.shape[0]
        return acc
    
    def save_params(self, file_name='params.pkl'):
        """
        パラメータをセーブするための関数
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def save_model(self, file_name='model.pkl'):
        """
        ニューラルネットのモデルをセーブする関数
        """
        model = {}
        for key, val in self.sequential.items():
            model[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)


    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        #for i, layer_idx in enumerate()

if __name__ == '__main__':
    import keras
    import numpy as np
    from keras.datasets import mnist
    from functions import Datafeature, train_test_splint
    from callbacks import LearningVisualizationCallback
    

    #訓練データの読み込み
    x_train = np.loadtxt(
        "./data/learn_1.csv",  # 読み込むファイル名(例"save_data.csv")
        dtype=float,  # データのtype
        delimiter=",",  # 区切り文字の指定
        ndmin=2  # 配列の最低次元
    )

    #テストデータの読み込み
    t_train = np.loadtxt(
        "./data/test_1.csv",  # 読み込むファイル名(例"save_data.csv")
        dtype=float,  # データのtype
        delimiter=",",  # 区切り文字の指定
        ndmin=2  # 配列の最低次元
    )

    x_val = np.loadtxt(
        "./data/val_l.csv",  # 読み込むファイル名(例"save_data.csv")
        dtype=float,  # データのtype
        delimiter=",",  # 区切り文字の指定
        ndmin=2  # 配列の最低次元
    )

    t_val = np.loadtxt(
        "./data/val_t.csv",  # 読み込むファイル名(例"save_data.csv")
        dtype=float,  # データのtype
        delimiter=",",  # 区切り文字の指定
        ndmin=2  # 配列の最低次元
    )
    
    feature = None
    #-------------------------------
    # DataFeature
    #-------------------------------
    if feature != None:
            x_train = Datafeature(x_train, feature)
            t_train = Datafeature(t_train, feature)
            x_val = Datafeature(x_val, feature)
            t_val = Datafeature(t_val, feature)
    
    # クロスバリエーション
    #x_train, x_test, t_train, t_test, x_val, t_val = \
        #train_test_splint(x_train, t_train, 1000, 100, random_state=1)
    
    """
    #データを読み込む
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    #データをfloat型に変換
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #0～255までの範囲のデータを0～1までの範囲に変更
    x_train /= 255
    x_test /= 255

    #Mnistデータを加工する
    x_train = x_train.reshape(60000, 784)  # 1次元配列に変換
    x_test = x_test.reshape(10000, 784)

    #正解データの加工
    t_train = keras.utils.to_categorical(t_train, 10)  # one_hot_labelに変換
    t_test = keras.utils.to_categorical(t_test,  10)
    """


    # 学習曲線を可視化するコールバックを用意する
    higher_better_metrics = ['r2']
    visualize_cb = LearningVisualizationCallback(higher_better_metrics)
    callbacks = [
        visualize_cb,
    ]

    model = Sequential()
    model.add(Input(input_shape=x_train.shape[1]))
    model.add(Dense(50, activation='relu', weight_initializer='relu'))
    model.add(Dense(50, activation='relu', weight_initializer='relu'))
    #model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
    #model.add(Dense(50, activation='sigmoid', weight_initializer='sigmoid'))
    #model.add(Dense(t.shape[1],  activation='softmax'))
    #model.compile(loss='cross_entropy_error')
    model.add(Dense(t_train.shape[1], activation = 'liner'))
    model.compile(loss='mean_squared_error',
                  optimizer='sgd', metrics=['r2'])
    
    epochs = 10
    batch_size = 128

    history = model.fit(x_train, t_train, batch_size=batch_size,
                        epochs=epochs, validation=None, callbacks=callbacks)

    #score = model.evaluate(x_test, t_test)

    output = model.flow([-50, 0, 3])

    model.save_params()
    model.save_model()
