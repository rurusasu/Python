import numpy as np

#出力層
#2乗和誤差レイヤ
class mean_squared_error:
    def __init__(self):
        self.loss = None #損失
        self.y    = None #linerの出力
        self.t    = None #教師データ

    def forward(self, x, t):
        self.y = x
        self.t = t
        if t.shape != x.shape:
            self.y = self.y.reshape(self.y.size, 1)
            self.t = self.t.reshape(self.t.size, 1)
        self.loss = 0.5 * np.sum((self.y - self.t)**2)

        return self.loss

    def backward(self, dout = 1):
        #if dout != 1:
        if dout == 1:
            batch_size = self.t.shape[0]
            if self.y.size == self.t.size:
                dout = (self.y - self.t) / batch_size

        return dout


#恒等関数レイヤ
class liner:
    def __init__(self):
        self.x = None 

    def forward(self, x):
        self.x = x
        return self.x

    def backward(self, dout):
        delta = 1 * dout

        return delta


#ReLUレイヤ
class relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0) #xが0以下ならFalseを返す(0より大きければTrue)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


#Affine
class affine:
    def __init__(self, W, b):
        self.W = W
        self.B = b

        self.x = None
        self.original_x_shape = None
        #重み・バイアスパラメータの微分
        self.dW = None
        self.dB = None

    def forward(self, x):
        #テンソル対応
        self.original_x_shape = x.shape #元の形を記憶させる
        x = x.reshape(x.shape[0], -1)   #奥行き方向の幅を固定しつつ、行列の大きさを変更
        self.x = x
        
        out = np.dot(self.x, self.W) + self.B

        return out

    def backward(self, dout):
        #printの設定
        #print('第%d層 - AffineLayer - Weight%d, %d' %(counter, counter-1, counter))
        #print(self.W)
        #print('第%d層 - AffineLayer - Bias%d' %(counter, counter))
        #print(self.B)

        dx = np.dot(dout, self.W.T)
        #self.W = self.W - np.dot(self.x.T, dout)
        #self.B = self.B - np.sum(dout, axis = 0)
        self.W = np.dot(self.x.T, dout)
        self.B = np.sum(dout, axis = 0)

        dx = dx.reshape(*self.original_x_shape) #逆伝播を入力信号の形に戻す
        return dx


class InputLayer:
    def __init__(self, input_shape):
        #self.Input_Row_Size = input_shape
        self.Input_Col_Size = None
        self.input          = None
        #
        #if len(input_shape) == 1:   #もし、入力数が配列で指定されたとき
            #self.input = 1
            #pass
        #elif len(input_shape) == 2:
            #self.input = input_shape[1]


    def unit(self, Data_Col_Size, counter):
        print('第%d層 - InputLayer' %counter)
        self.Input_Col_Size = Data_Col_Size
    
        return self.Input_Col_Size

    def forward(self, input_data):
        y = np.reshape(input_data, [-1, self.Input_Col_Size])

        return out

    def backward(self, dout):
        pass


    #データコピー
def data_copy(data):
    dim = data.ndim                      #受け取ったdataの次元を確認

    if dim == 1 :                        #dataが1次元(配列)のとき
        return data
    else :                               #dataが2次元(行列)のとき
        data_copy = np.empty_like(data)  #dataと同じ大きさの空の行列を作成
        col = data.shape[1]              #dataの列数を取得

        for i in range(col):
            data_copy[:, i] = data[:, i] #対応する列に値をコピー
        
        return data_copy

#標準化
def data_std(data):
    dim = data.ndim #受け取ったdataの次元を確認

    if dim == 1 :   #dataが1次元(配列)のとき
        return (data - data.mean()) / data.std()
    else :                               #dataが2次元(行列)のとき
        data_std = np.empty_like(data)   #dataと同じ大きさの空の行列を作成
        col = data.shape[1]              #dataの列数を取得

        for i in range(col):
            data_std[:, i] =(data[:, i] - data[:, i].mean()) / data[:, i].std() #対応する列を標準化
        
        return data_std


#正規化
def data_nom(data):
    dim = data.ndim #受け取ったdataの次元を確認

    if dim == 1 :   #dataが1次元(配列)のとき
        return data_1[:, 0] / max(abs(data_1[:, 0]))
    else :                               #dataが2次元(行列)のとき
        data_nom = np.empty_like(data)   #dataと同じ大きさの空の行列を作成
        col = data.shape[1]              #dataの列数を取得

        for i in range(col):
            data_nom[:, i] =data[:, i] / max(abs(data[:, i])) #対応する列を正規化
        
        return data_nom


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
data = data_copy(data)
test = data_copy(test)

#標準化
data_1 = data_std(data_1)
test_1 = data_std(test_1)

#正規化
data_ = data_nom(data_)
test_ = data_nom(test_)

#訓練データのセット
x_train = data_[:, 0:2] #入力データをセット
t_train = data_[:, 2]   #正解データをセット

#テストデータのセット
x_test  = test_[:, 0:2] #入力データをセット
t_test  = test_[:, 2]   #正解データをセット


ACTIVATION = 1 #0:Sigmoid function, 1:ReLU function
ReLU_GAIN  = 0.7

if ACTIVATION == 1:
    #for ReLU
    eta      = 0.0001 #重みの学習係数(learning rate for weights),ReLUの場合、値を大きくすると誤差が減少しない。
    beta     = 0.0001 #閾値の学習係数(learning rate for bias)(活性化関数を入力軸上で微小移動)
    eta_myu  = 0.01   #慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す
    beta_myu = 0.01   #慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す

else:
    #for Sigmoid
    eta      = 0.1 #重みの学習係数(learning rate for weights),ReLUの場合、値を大きくすると誤差が減少しない。
    beta     = 0.1 #閾値の学習係数(learning rate for bias)(活性化関数を入力軸上で微小移動)
    eta_myu  = 0.1 #慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す
    beta_myu = 0.1 #慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す




Batch_size = x_train.shape[0]
Iteration_limit = 20 # epoche回数
Minibatch_size = 128

#各レイヤごとの層数
N1 = 2
N2 = 50
N3 = 50
N4 = 50
N5 = 1



#重みの初期化
w1 = np.random.randn(N1, N2) / np.sqrt(N1) 
w2 = np.random.randn(N2, N3) / np.sqrt(N2)
w3 = np.random.randn(N3, N4) / np.sqrt(N3)
w4 = np.random.randn(N4, N5) / np.sqrt(N4)

#閾値の初期化
b2 = np.zeros(N2) 
b3 = np.zeros(N3)
b4 = np.zeros(N4)
b5 = np.zeros(N5)

#最初の重みの保存
w1_1 = w1
w2_1 = w2
w3_1 = w3
w4_1 = w4

#最初の閾値の保存
b2_1 = b2
b3_1 = b3
b4_1 = b4
b5_1 = b5

#状態ベクトルの初期化
s1 = np.zeros((Minibatch_size, N1))
s2 = np.zeros((Minibatch_size, N2))
s3 = np.zeros((Minibatch_size, N3))
s4 = np.zeros((Minibatch_size, N4))
s5 = np.zeros((Minibatch_size, N5))

#出力ベクトルの初期化
x1 = np.zeros((Minibatch_size, N1))
x2 = np.zeros((Minibatch_size, N2))
x3 = np.zeros((Minibatch_size, N3))
x4 = np.zeros((Minibatch_size, N4))
x5 = np.zeros((Minibatch_size, N5))


'''
学習の進捗状況（訓練データ内の１サンプルあたりの誤差）を保存するバッファ
誤差は目標出力とNNからの出力との差
'''
Ev_buff = np.zeros((Iteration_limit, 1))

for i in range(Iteration_limit):
    #行数からMinibatch_sizeだけランダムに値を抽出 replace(重複)
    Batch_mask = np.random.choice(Batch_size, Minibatch_size, replace = False)
    train_data = x_train[Batch_mask, 0:2]  #訓練データをセット
    test_data =  t_train[Batch_mask]       #全訓練データからBatch_maskだけ抽出

    #抽出された訓練データの行数と列数を取得
    Row    = data_.shape[0]
    Column = data_.shape[1]
    #標準化と正規化されたデータ保存用
    x5_buff     = np.zeros((Row, N5)) #NNからの出力（第5層のユニットからの出力）の保存先を確保
    x5_err_buff = np.zeros((Row, N5)) #NNからの出力（第5層のユニットからの出力）と訓練データとの誤差の保存先を確保
    #標準化と正規化される前のオリジナルデータ保存用
    x5_buff2 = np.zeros((Row, N5))
    x5_err_buff2 = np.zeros((Row, N5))
    for sample_counter in range(Minibatch_size):
        #第一層
        s1 = train_data
        x1 = s1 #入力層ではそのまま出力される
        #第二層
        s2 = np.zeros((Minibatch_size, N2)) #状態量を求めるために初期化する。
        s2 = s2 + np.dot(x1, w1) #第二層の状態ベクトルの成分を求める。
        s2 = s2 + b2 #状態量に閾値を加える。-> 活性化関数への入力となる。
        if ACTIVATION == 0:
            x2 = (1 / (1 + np.exp(-s2))) - 0.5
        else:
            mask = (x2 <= 0) #xが0以下ならFalseを返す(0より大きければTrue)
            x2[mask] = ReLU_GAIN * s2[mask]
        #第三層
        s3 = np.zeros((Minibatch_size, N3)) #状態量を求めるために初期化する。
        s3 = s3 + np.dot(x2, w2) #第二層の状態ベクトルの成分を求める。
        s3 = s3 + b3 #状態量に閾値を加える。-> 活性化関数への入力となる。
        if ACTIVATION == 0:
            x3 = (1 / (1 + np.exp(-s3))) - 0.5
        else:
            mask = (x3 <= 0) #xが0以下ならFalseを返す(0より大きければTrue)
            x3[mask] = ReLU_GAIN * s3[mask]
        #第四層
        s4 = np.zeros((Minibatch_size, N4)) #状態量を求めるために初期化する。
        s4 = s4 + np.dot(x3, w3) #第二層の状態ベクトルの成分を求める。
        s4 = s4 + b4 #状態量に閾値を加える。-> 活性化関数への入力となる。
        if ACTIVATION == 0:
            x4 = (1 / (1 + np.exp(-s4))) - 0.5
        else:
            mask = (x4 <= 0) #xが0以下ならFalseを返す(0より大きければTrue)
            x4[mask] = ReLU_GAIN * s4[mask]
        #第五層
        s5 = np.zeros((Minibatch_size, N5)) #状態量を求めるために初期化する。
        s5 = s5 + np.dot(x4, w4) #第二層の状態ベクトルの成分を求める。
        s5 = s5 + b5 #状態量に閾値を加える。-> 活性化関数への入力となる。
        x5 = s5

        x5_d = train_data
        #標準化と正規化処理されたデータによる出力と誤差
        x5_buff = x5 #NNからの出力を保存しているだけ。なくてもNNの学習はできる。
        x5_err_buff = 0.5*(x5 - x5_d) ** 2
        #オリジナルデータとの出力誤差
        #for i in range(Row):
            #x5_buff2[i, j] = data[i, j] * max(abs(data[i, j])) * data[i, j].std() + data[i, j].mean()  #対応する列を正規化
        [w1, w2, w3, w4, b2, b3, b4, b5, w1_1, w2_1, w3_1, w4_1, b2_1, b3_1, b4_1, b5_1] = 


def weight(w1, w2, w3, w4, th2, th3, th4, th5, eta, beta, x1, x2, x3, x4, x5, x5_d, s2, s3, s4, s5, w1_1, w2_1, w3_1, w4_1, eta_myu, N1, N2, N3, N4, N5, th2_1, th3_1, th4_1, th5_1, ACTIVATION, ReLU_GAIN):
    #前回の重みを保存しておく(入力層と第２層（第１隠れ層）間の重み)
    w1_tmp = w1_1
    w2_tmp = w2_1 #第２層（第１隠れ層）と第３層（第２隠れ層）間の重み
    w3_tmp = w3_1 #第３層（第２隠れ層）と第４層（第３隠れ層）間の重み
    w4_tmp = w4_1 #第４層（第３隠れ層）と第５層（出力層）間の重み
    #前回の閾値を保存しておく
    th2_tmp = th2_1
    th3_tmp = th3_1
    th4_tmp = th4_1
    th5_tmp = th5_1
    #慣性項の計算のために現在の重みを前回の重みとして保存する。
    w1_1 = w1
    w2_1 = w2
    w3_1 = w3
    w4_1 = w4
    #慣性項の計算のために、現在の閾値を前回の閾値として保存する
    th2_1 = th2
    th3_1 = th3
    th4_1 = th4
    th5_1 = th5
    delta5 = np.zeros((Minibatch_size, N5))
    delta4 = np.zeros((Minibatch_size, N4))
    delta3 = np.zeros((Minibatch_size, N3))
    delta2 = np.zeros((Minibatch_size, N2))


#a1 = affine(W1, B1)
#r1 = relu()
#a2 = affine(W2, B2)
#r2 = relu()
#l = liner()
#loss = mean_squared_error()

'''
for i in range(20):
    x = a1.forward(x)
    x = r1.forward(x)
    x = a2.forward(x)
    x = r2.forward(x)
    x = l.forward(x)
    x = loss.forward(x, t)

    x = l.backward(x)
    x = r2.backward(x)
    x = a2.backward(x)
    x = r1.backward(x)
    x = a1.backward(x)

print(x)
'''