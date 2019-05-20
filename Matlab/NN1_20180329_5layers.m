%このNNは、二次曲面z=ax^2+by^2+cxy+dx+ey+f を近似できるように学習されます。
%学習のための訓練データ(x,y,z)はtrainingset1で事前に作っておきます。
%sigmoid.mではシグモイド関数を定義しグラフの形状を確認した
clear all;%ワーク領域をクリア
format long;%longフォーマットで数値を表示する
load NN1_data.mat;%事前に作成しておいた訓練データ（data）の読み込み
ACTIVATION = 1 ;%0:Sigmoid function, 1:ReLU function
ReLU_GAIN=0.7;
%学習データの標準化（standardization）:平均が0の分散が1となるように入力データのシフティングとスケーリングを行います。 
data_1(:,1:1)=(data(:,1:1)-mean(data(:,1:1)))/std(data(:,1:1));%教師データの標準化処理(学習効率UPに非常に重要！)
data_1(:,2:2)=(data(:,2:2)-mean(data(:,2:2)))/std(data(:,2:2));
data_1(:,3:3)=(data(:,3:3)-mean(data(:,3:3)))/std(data(:,3:3));
%data_1には標準化された値が格納されている。
%標準化された学習データをもとに、正規化処理（normalization）を行う
data_(:,1:1)=data_1(:,1:1)/max(abs(data_1(:,1:1)));%教師データの正規化処理(学習効率UPに非常に重要！)
data_(:,2:2)=data_1(:,2:2)/max(abs(data_1(:,2:2)));
data_(:,3:3)=data_1(:,3:3)/max(abs(data_1(:,3:3)));
%data_には更に正規化された値が格納されている。
%Batch Normalization とは？
data_tmp = data_;%ミニバッチ処理で使用するために、標準化と正規化が行われたdata_をdata_tmpに保存
if ACTIVATION ==1
    % for ReLU
    eta=0.0001; %重みの学習係数(learning rate for weights),ReLUの場合、値を大きくすると誤差が減少しない。
    beta=0.0001; %閾値の学習係数(learning rate for threshold)(活性化関数を入力軸上で微小移動)
    eta_myu=0.01;%慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す
    beta_myu=0.01;%慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す
else
    % for Sigmoid
    eta=0.1; %重みの学習係数(learning rate for weights),ReLUの場合、値を大きくすると誤差が減少しない。
    beta=0.1; %閾値の学習係数(learning rate for threshold)(活性化関数を入力軸上で微小移動)
    eta_myu=0.1;%慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す
    beta_myu=0.1;%慣性項（inertia）とは前回の重みの更新量を反映させる程度を表す
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%myu=0.3;%慣性項(inertia)とは前回の重みの更新量を反映させる程度、前回の情報を使う
N1 = 2 ;%入力の数（x,y）
N2 = 50 ;%第２層（第１隠れ層）のニューロン(ユニット)の数
N3 = 50 ;%第３層（第２隠れ層）のニューロンの数
N4 = 50 ;%第４層（第３隠れ層）のニューロンの数
N5 = 1 ;%出力の数（z）
%
K=2;
w1 = K*(ones(N1,N2)*0.5 - rand(N1,N2)) ;%-1から+1の範囲で重みの初期化
w2 = K*(ones(N2,N3)*0.5 - rand(N2,N3)) ;%重みの初期化
w3 = K*(ones(N3,N4)*0.5 - rand(N3,N4)) ;%重みの初期化
w4 = K*(ones(N4,N5)*0.5 - rand(N4,N5)) ;%重みの初期化
%
th2 = K*(ones(N2,1)*0.5 - rand(N2,1)) ;%閾値の初期化
th3 = K*(ones(N3,1)*0.5 - rand(N3,1)) ;%閾値の初期化
th4 = K*(ones(N4,1)*0.5 - rand(N4,1)) ;%閾値の初期化
th5 = K*(ones(N5,1)*0.5 - rand(N5,1)) ;%閾値の初期化
%
w1_1 = w1 ;%最初の重みの保存
w2_1 = w2 ;%最初の重みの保存
w3_1 = w3 ;%最初の重みの保存
w4_1 = w4 ;%最初の重みの保存
%
th2_1 = th2 ;%最初の閾値の保存
th3_1 = th3 ;%最初の閾値の保存
th4_1 = th4 ;%最初の閾値の保存
th5_1 = th5 ;%最初の閾値の保存(未使用？)
%
s1 = zeros(N1,1) ;%第1層のニューロンの状態ベクトルの初期化（0でクリアした保存先を確保）
x1 = zeros(N1,1) ;%第1層のニューロンの出力ベクトルの初期化
s2 = zeros(N2,1) ;%第2層のニューロンの状態ベクトルの初期化
x2 = zeros(N2,1) ;%第2層のニューロンの出力ベクトルの初期化
s3 = zeros(N3,1) ;%第3層のニューロンの状態ベクトルの初期化
x3 = zeros(N3,1) ;%第3層のニューロンの出力ベクトルの初期化
s4 = zeros(N4,1) ;%第4層のニューロンの状態ベクトルの初期化
x4 = zeros(N4,1) ;%第4層のニューロンの出力ベクトルの初期化
s5 = zeros(N5,1) ;%第5層のニューロンの状態ベクトルの初期化
x5 = zeros(N5,1) ;%第5層のニューロンの出力ベクトルの初期化
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%デルタという量が何か確認せよ(BP法で使用される)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BP法で使用されるデルタという量が何かは確認する必要がある。
%delta5 = zeros(N5,1) ;%BP法で必要となるそれぞれのニューロンのデルタと呼ばれる値の保存先を確保
%delta4 = zeros(N4,1) ;%BP法で必要となるそれぞれのニューロンのデルタと呼ばれる値の保存先を確保
%delta3 = zeros(N3,1) ;%BP法で必要となるそれぞれのニューロンのデルタと呼ばれる値の保存先を確保
%delta2 = zeros(N2,1) ;%BP法で必要となるそれぞれのニューロンのデルタと呼ばれる値の保存先を確保
%delta1 = zeros(N1,1) ;%BP法で必要となるそれぞれのニューロンのデルタと呼ばれる値の保存先を確保
Batch_size = size(data(:,1:1));%訓練データ内のサンプル数をBatch_sizeに格納
Iteration_limit = 1000000;%Minibatch_sizeのデータを使った最大学習回数
%Minibatch_size = Batch_size(1,1);
Minibatch_size = 10000;%Batch_sizeから抽出されるサイズ
Ev_buff = zeros(Iteration_limit,1);%学習の進捗状況（訓練データ内の１サンプルあたりの誤差）を保存するバッファであり、誤差は目標出力とNNからの出力との差である。
for iteration = 1:Iteration_limit %全ての訓練データを使った前向き計算とBP法による後ろ向き計算を、Max_epochの回数だけ行う。
    data_=data_tmp;%標準化と正規化がされた訓練データをセット
    data_ = datasample(data_(:,:),Minibatch_size);%全訓練データからMinibatch_sizeだけ抽出
    [Row Column] = size(data_);%抽出された訓練データの行数と列数を取得
    %標準化と正規化されたデータ保存用
    x5_buff = zeros(Row,N5) ;%NNからの出力（第5層のユニットからの出力）を保存していくバッファをクリア
    x5_err_buff = zeros(Row,N5) ;%NNからの出力（第5層のユニットからの出力）と訓練データとの誤差を保存していくバッファをクリア
    %標準化と正規化される前のオリジナルデータ保存用
    x5_buff2 = zeros(Row,N5) ;%NNからの出力（第5層のユニットからの出力）を保存していくバッファをクリア
    x5_err_buff2 = zeros(Row,N5) ;%NNからの出力（第5層のユニットからの出力）と訓練データとの誤差を保存していくバッファをクリア
    for sample_counter = 1:Row %Rowの回数（Minibatch_sizeで指定された訓練データ）だけ前向き計算とBP法による後ろ向き計算（重みの更新）を行う。
        %第一層
        s1(1)=data_(sample_counter:sample_counter,1);%sample_counterは訓練データの第何行かを表す。
        s1(2)=data_(sample_counter:sample_counter,2);
        x1(1)=s1(1);%入力層ではそのまま出力される
        x1(2)=s1(2);
        %第二層     
        for i=1:N2%iは第2層のユニット数
            s2(i) = 0 ;%状態量を求めるためにまずは初期化する必要がある。
            for j = 1:N1%jは第1層のユニット数
                s2(i) = s2(i) + w1(j,i)*x1(j) ;%第2層の状態ベクトルの成分を求める。
            end
            s2(i) = s2(i) + th2(i) ;%状態量に閾値を加える。->　活性化関数への入力となる。
            %
            if ACTIVATION == 0
                x2(i) = 1/(1+exp(-s2(i)))-1/2 ;%活性化関数（シグモイド関数）からの出力を求める。
            else
                if s2(i) > 0                    %活性化関数（ReLU）からの出力を求める。
                    x2(i) = s2(i);
                else
                    x2(i) = ReLU_GAIN*s2(i);
                end
            end
        end
        %x2= x2/max(abs(x2));
        %第三層
        for i=1:N3%iは第3層のユニット数
            s3(i) = 0 ;%状態量を求めるためにまずは初期化する必要がある。
            for j = 1:N2%jは第2層のユニット数
                s3(i) = s3(i) + w2(j,i)*x2(j) ;%第3層の状態ベクトルの成分を求める。
            end
            s3(i) = s3(i) + th3(i) ;%状態量に閾値を加える。->　活性化関数への入力となる。
            %
            if ACTIVATION == 0
                x3(i) = 1/(1+exp(-s3(i))) -1/2;%活性化関数からの出力を求める。
            else
                if s3(i) > 0
                    x3(i) = s3(i);
                else
                    x3(i) = ReLU_GAIN*s3(i);
                end
            end
        end  
        %x3= x3/max(abs(x3));
        %第四層     
        for i=1:N4%iは第4層のユニット数
            s4(i) = 0 ;%状態量を求めるためにまずは初期化する必要がある。
            for j = 1:N3%jは第3層のユニット数
                s4(i) = s4(i) + w3(j,i)*x3(j) ;%第4層の状態ベクトルの成分を求める。
            end
            s4(i) = s4(i) + th4(i) ;%状態量に閾値を加える。
            %
            if ACTIVATION == 0
                x4(i) = 1/(1+exp(-s4(i))) -1/2;%活性化関数からの出力を求める。
             else
                 if s4(i) > 0
                     x4(i) = s4(i);
                 else
                     x4(i) = ReLU_GAIN*s4(i);
                 end
             end
        end
        %第五層     
        for i=1:N5%iは第5層のユニット数
            s5(i) = 0 ;%状態量を求めるためにまずは初期化する必要がある。
            for j = 1:N4%jは第4層のユニット数
                s5(i) = s5(i) + w4(j,i)*x4(j) ;%第5層の状態ベクトルの成分を求める。
            end
            x5(i)=s5(i);
            %x5(i) = 1/(1+exp(-s5(i))) -1/2;%活性化関数からの出力を求める。            
        end
        x5_d=data_(sample_counter,3);%BP法では、x5がx5_d(事前に用意し、標準化、正規化された訓練データ)に近づくように重みが調整される。
        for i=1:N5
            %標準化と正規化処理されたデータによる出力と誤差
            x5_buff(sample_counter,i)=x5(i);%NNからの出力を保存しているだけ。なくてもNNの学習はできます。
            %x5_err_buff(sample_counter,i)=abs(x5_d-x5(i));%NNからの出力と訓練データとの誤差を保存しておく
            x5_err_buff(sample_counter,i)=0.5*(x5(i)-x5_d)^2;
            %オリジナルのデータによる出力と誤差
            x5_buff2(sample_counter,i)=x5(i)*max(abs(data_1(:,3:3)))*std(data(:,3:3))+mean(data(:,3:3));
            x5_err_buff2(sample_counter,i)=abs(data(sample_counter,3)-x5_buff2(sample_counter,i));
        end
        %関数weightでは、BP法により重みが更新されます。x5とx5_dで誤差が作られます。BP法では一般化デルタルールが適用されます。
        %一般的に、[戻り値]　＝　関数（引数）；の形式となる。
        [w1,w2,w3,w4,th2,th3,th4,th5,w1_1,w2_1,w3_1,w4_1,th2_1,th3_1,th4_1,th5_1] = weight_20180329_5layers(w1,w2,w3,w4,th2,th3,th4,th5,eta,beta,x1,x2,x3,x4,x5,x5_d,s2,s3,s4,s5,w1_1,w2_1,w3_1,w4_1,eta_myu,beta_myu,N1,N2,N3,N4,N5,th2_1,th3_1,th4_1,th5_1,ACTIVATION,ReLU_GAIN) ;
    end
    err_sum =0;
    for i=1:sample_counter
        for j=1:N5
            err_sum =err_sum +x5_err_buff(i,j);%Minibatch_sizeで指定されたデータ分の誤差の総和
        end
    end
    Ev_buff(iteration) = err_sum/sample_counter;%訓練データ内の１サンプルあたりの平均誤差を保存
    if rem(iteration,1)==0%ここで表示頻度を設定している。
        Epoch = round(Minibatch_size*iteration/Batch_size(1,1));
        fprintf('\n Epock = %d, Iteration = %g', Epoch,iteration);%現在のミニバッチ実行回数を表示 
        fprintf('\n err_sum/sample_counter = %g', err_sum/sample_counter); 
        plot(Ev_buff(1:iteration));%学習の進捗状況を表示する
        pause(0.1);%グラフ表示を確実に行うための0.1秒間のポーズ
        save NN1_weight w1 w2 w3 w4 th2 th3 th4 th5 Ev_buff N1 N2 N3 N4 N5 eta beta eta_myu beta_myu iteration Minibatch_size ACTIVATION ;%学習した重みとパラメータなどを保存する。
    end
    if err_sum/sample_counter < 0.0001 %平均誤差が小さくなったら学習の終了です。　
        save NN1_weight w1 w2 w3 w4 th2 th3 th4 th5 Ev_buff N1 N2 N3 N4 N5 eta beta eta_myu beta_myu iteration Minibatch_size ACTIVATION ;%学習した重みとパラメータなどを保存する。
        break;
    end
end


