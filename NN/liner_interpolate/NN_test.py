import numpy as np
import matplotlib.pyplot as plt

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


data = np.loadtxt(
    "save_data.csv", #読み込むファイル名(例"save_data.csv")
    dtype=float,     #データのtype
    delimiter=",",   #区切り文字の指定
    ndmin=2          #配列の最低次元
    )

#読み込んだデータを学習用にコピーする
data_1 = data_copy(data)

#標準化
data_ = data_std(data_1)

#正規化
data_ = data_nom(data_)

data_tmp = data_ #ミニバッチ処理で使用するために、標準化と正規化が行われたdata_をdata_tmpに保存


N1 = 2 #入力の数(x, y)
N2 = 50 #第2層(第1隠れ層)のニューロン(ユニット)の数
N3 = 50 #第3層(第2隠れ層)のニューロンの数
N4 = 50 #第4層(第3隠れ層)のニューロンの数
N5 = 1  #出力層

K=2;
w1 = K*(np.ones((N1,N2))*0.5 - np.random.rand(N1,N2)) #-1から+1の範囲で重みの初期化
w2 = K*(np.ones((N2,N3))*0.5 - np.random.rand(N2,N3)) #重みの初期化
w3 = K*(np.ones((N3,N4))*0.5 - np.random.rand(N3,N4)) #重みの初期化
w4 = K*(np.ones((N4,N5))*0.5 - np.random.rand(N4,N5)) #重みの初期化

th2 = K*(np.ones((N2,1))*0.5 - np.random.rand(N2,1))  #閾値の初期化
th3 = K*(np.ones((N3,1))*0.5 - np.random.rand(N3,1))  #閾値の初期化
th4 = K*(np.ones((N4,1))*0.5 - np.random.rand(N4,1))  #閾値の初期化
th5 = K*(np.ones((N5,1))*0.5 - np.random.rand(N5,1))  #閾値の初期化
#
w1_1 = w1 #最初の重みの保存
w2_1 = w2 #最初の重みの保存
w3_1 = w3 #最初の重みの保存
w4_1 = w4 #最初の重みの保存
#
th2_1 = th2 #最初の閾値の保存
th3_1 = th3 #最初の閾値の保存
th4_1 = th4 #最初の閾値の保存
th5_1 = th5 #最初の閾値の保存

s1 = np.zeros((N1,1)) #第1層のニューロンの状態ベクトルの初期化（0でクリアした保存先を確保）
x1 = np.zeros((N1,1)) #第1層のニューロンの出力ベクトルの初期化
s2 = np.zeros((N2,1)) #第2層のニューロンの状態ベクトルの初期化
x2 = np.zeros((N2,1)) #第2層のニューロンの出力ベクトルの初期化
s3 = np.zeros((N3,1)) #第3層のニューロンの状態ベクトルの初期化
x3 = np.zeros((N3,1)) #第3層のニューロンの出力ベクトルの初期化
s4 = np.zeros((N4,1)) #第4層のニューロンの状態ベクトルの初期化
x4 = np.zeros((N4,1)) #第4層のニューロンの出力ベクトルの初期化
s5 = np.zeros((N5,1)) #第5層のニューロンの状態ベクトルの初期化
x5 = np.zeros((N5,1)) #第5層のニューロンの出力ベクトルの初期化

Batch_size = data_1.shape[1] #訓練データ内のサンプル数をBatch_sizeに格納
Iteration_limit = 1000000     #Minibatch_sizeのデータを使った最大学習回数
#Minibatch_size = Batch_size(1,1);
Minibatch_size = 10000        #Batch_sizeから抽出されるサイズ
Ev_buff = np.zeros((Iteration_limit,1))  #学習の進捗状況（訓練データ内の１サンプルあたりの誤差）を保存するバッファであり、誤差は目標出力とNNからの出力との差である。

for iteration in range(Iteration_limit): #全ての訓練データを使った前向き計算とBP法による後ろ向き計算を、Max_epochの回数だけ行う。
    data_=data_tmp #標準化と正規化がされた訓練データをセット
    data_ = np.random.choice(data[:, 0:2] ,Minibatch_size)  #全訓練データからMinibatch_sizeだけ抽出

    Row, Colum = data_.shape([0]), data.shape([1]) #抽出された訓練データの行数と列数を取得
    #標準化と正規化されたデータ保存用
    x5_buff = np.zeros(Row,N5)      #NNからの出力（第5層のユニットからの出力）を保存していくバッファをクリア
    x5_err_buff = np.zeros(Row,N5)  #NNからの出力（第5層のユニットからの出力）と訓練データとの誤差を保存していくバッファをクリア
    #標準化と正規化される前のオリジナルデータ保存用
    x5_buff2 = np.zeros(Row,N5)      #NNからの出力（第5層のユニットからの出力）を保存していくバッファをクリア
    x5_err_buff2 = np.zeros(Row,N5)  #NNからの出力（第5層のユニットからの出力）と訓練データとの誤差を保存していくバッファをクリア
   
    for sample_counter in range(Row):    #Rowの回数（Minibatch_sizeで指定された訓練データ）だけ前向き計算とBP法による後ろ向き計算（重みの更新）を行う。
        #第一層
        s1[1]=data_[sample_counter:sample_counter,1] #sample_counterは訓練データの第何行かを表す。
        s1[2]=data_[sample_counter:sample_counter,2]
        x1[1]=s1[1] #入力層ではそのまま出力される
        x1[2]=s1[2]
        #第二層     
        for i in range(N2): #iは第2層のユニット数
            s2[i] = 0 #状態量を求めるためにまずは初期化する必要がある。
            for j in range(N1): #jは第1層のユニット数
                s2[i] = s2[i] + w1[j,i]*x1[j] #第2層の状態ベクトルの成分を求める。

            s2[i] = s2[i] + th2[i] #状態量に閾値を加える。->　活性化関数への入力となる。

            if ACTIVATION == 0:
                x2[i] = 1/(1+exp(-s2[i]))-1/2 #活性化関数（シグモイド関数）からの出力を求める。
            else:
                if s2[i] > 0:                  #活性化関数（ReLU）からの出力を求める。
                    x2[i] = s2[i]
                else:
                    x2[i] = ReLU_GAIN*s2[i]

        #第三層
        for i in range(N3): #iは第3層のユニット数
            s3[i] = 0 #状態量を求めるためにまずは初期化する必要がある。
            for j in range(N2): #jは第2層のユニット数
                s3[i] = s3[i] + w2[j,i]*x2[j] #第3層の状態ベクトルの成分を求める。

            s3[i] = s3[i] + th3[i] #状態量に閾値を加える。->　活性化関数への入力となる。

            if ACTIVATION == 0:
                x3[i] = 1/(1+exp(-s3[i])) -1/2 #活性化関数からの出力を求める。
            else:
                if s3[i] > 0:
                    x3[i] = s3[i]
                else:
                    x3[i] = ReLU_GAIN*s3[i]

        #第四層     
        for i in range(N4): #iは第4層のユニット数
            s4[i] = 0 #状態量を求めるためにまずは初期化する必要がある。
            for j in range(N3): #jは第3層のユニット数
                s4[i] = s4[i] + w3[j,i]*x3[j] #第4層の状態ベクトルの成分を求める。

            s4[i] = s4[i] + th4[i] #状態量に閾値を加える。

            if ACTIVATION == 0:
                x4[i] = 1/(1+exp(-s4[i])) -1/2 #活性化関数からの出力を求める。
            else:
                 if s4[i] > 0:
                     x4[i] = s4[i]
                 else:
                     x4[i] = ReLU_GAIN*s4[i]
        #第五層     
        for i in range(N5): #iは第5層のユニット数
            s5[i] = 0 #状態量を求めるためにまずは初期化する必要がある。
            for j in range(N4): #jは第4層のユニット数
                s5[i] = s5[i] + w4[j,i]*x4[j] #第5層の状態ベクトルの成分を求める。
       
            x5[i] = s5[i]
        x5_d = data_[sample_counter,3]  #BP法では、x5がx5_d(事前に用意し、標準化、正規化された訓練データ)に近づくように重みが調整される。

        for i in range(N5):
            #標準化と正規化処理されたデータによる出力と誤差
            x5_buff[sample_counter,i] = x5[i] #NNからの出力を保存しているだけ。なくてもNNの学習はできます。
            x5_err_buff[sample_counter,i] = abs(x5_d - x5[i]) #NNからの出力と訓練データとの誤差を保存しておく
            #オリジナルのデータによる出力と誤差
            x5_buff2[sample_counter,i] = x5[i] * max(abs(data_1[:,3])) * std(data[:,3]) + mean(data[:,3]);
            x5_err_buff2[sample_counter,i] = abs(data[sample_counter,3] - x5_buff2[sample_counter,i])

        #関数weightでは、BP法により重みが更新されます。x5とx5_dで誤差が作られます。BP法では一般化デルタルールが適用されます。
        #一般的に、[戻り値]　＝　関数（引数）；の形式となる。
        w1, w2, w3, w4, th2, th3, th4, th5, w1_1, w2_1, w3_1, w4_1, th2_1, th3_1, th4_1, th5_1 = w1, w2, w3, w4, th2, th3, th4, th5, eta, beta, x1, x2, x3, x4, x5, x5_d, s2, s3, s4, s5, w1_1, w2_1, w3_1, w4_1, eta_myu, beta_myu, N1, N2, N3, N4, N5, th2_1, th3_1, th4_1, th5_1, ACTIVATION, ReLU_GAIN

        err_sum =0;
        for i in range(sample_counter):
            for j in range(N5):
                err_sum =err_sum +x5_err_buff(i,j) #Minibatch_sizeで指定されたデータ分の誤差の総和

        Ev_buff[iteration] = err_sum / sample_counter #訓練データ内の１サンプルあたりの平均誤差を保存
        if iteration % 1==0: #ここで表示頻度を設定している。
            Epoch = round(Minibatch_size*iteration / Batch_size)
            fprintf('\n Epock = %d, Iteration = %g', Epoch,iteration) #現在のミニバッチ実行回数を表示 
            fprintf('\n err_sum/sample_counter = %g', err_sum/sample_counter); 
            plt.plot(Ev_buff[1:iteration]) #学習の進捗状況を表示する
            pause(0.1) #グラフ表示を確実に行うための0.1秒間のポーズ

