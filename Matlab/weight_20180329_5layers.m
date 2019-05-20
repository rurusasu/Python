%バックプロパゲーションアルゴリズムにより重みを訓練する関数である。
function [w1,w2,w3,w4,th2,th3,th4,th5,w1_1,w2_1,w3_1,w4_1,th2_1,th3_1,th4_1,th5_1] = weight(w1,w2,w3,w4,th2,th3,th4,th5,eta,beta,x1,x2,x3,x4,x5,x5_d,s2,s3,s4,s5,w1_1,w2_1,w3_1,w4_1,eta_myu,beta_myu,N1,N2,N3,N4,N5,th2_1,th3_1,th4_1,th5_1,ACTIVATION,ReLU_GAIN) ;

    w1_tmp = w1_1 ;%前回の重みを保存しておく(入力層と第２層（第１隠れ層）間の重み)
	w2_tmp = w2_1 ;%第２層（第１隠れ層）と第３層（第２隠れ層）間の重み
	w3_tmp = w3_1 ;%第３層（第２隠れ層）と第４層（第３隠れ層）間の重み
	w4_tmp = w4_1 ;%第４層（第３隠れ層）と第５層（出力層）間の重み
    th2_tmp = th2_1;%前回の閾値を保存しておく
    th3_tmp = th3_1;
    th4_tmp = th4_1;
    th5_tmp = th5_1;
    w1_1 = w1 ;%慣性項の計算のために、現在の重みを前回の重みとして保存 (入力層と第２層（第１隠れ層）間の重み)する
	w2_1 = w2 ;
	w3_1 = w3 ;
	w4_1 = w4 ;
    th2_1 = th2;%慣性項の計算のために、現在の閾値を前回の閾値として保存する
    th3_1 = th3;
    th4_1 = th4;
    th5_1 = th5;
    delta5 = zeros(N5,1) ;%BP法で必要となるそれぞれのニューロンのデルタと呼ばれる値の保存先を確保
    delta4 = zeros(N4,1) ;%BP法で必要となるそれぞれのニューロンのデルタと呼ばれる値の保存先を確保
    delta3 = zeros(N3,1) ;%BP法で必要となるそれぞれのニューロンのデルタと呼ばれる値の保存先を確保
    delta2 = zeros(N2,1) ;%BP法で必要となるそれぞれのニューロンのデルタと呼ばれる値の保存先を確保
    %delta1 = zeros(N1,1) ;%BP法で必要となるそれぞれのニューロンのデルタと呼ばれる値の保存先を確保    
%%%
%%%	ステップ1：出力層に向かう結合係数(w4)を修正する。
%%%
   for i=1:N5
        if ACTIVATION == 0
            %シグモイド関数の場合 s5(第五層の出力(Affine + liner))
            delta5(i) = (1/(1+exp(-s5(i))))*(1-1/(1+exp(-s5(i))))*(x5(i) - x5_d(i)) ;%出力層におけるデルタ
        else
            %ReLUの場合
            if s5(i) > 0
                delta5(i)=(x5(i) - x5_d(i));
            else
                delta5(i) = ReLU_GAIN*(x5(i) - x5_d(i));
            end
        end
        for j = 1:N4
            w4(j,i) = w4(j,i) - eta * delta5(i)*x4(j) + eta_myu * (w4(j,i) - w4_tmp(j,i)) ;%慣性項ありの重み更新
        end
        th5(i) = th5(i) - beta * delta5(i)*x4(j) + beta_myu * (th5(i) - th5_tmp(i));%慣性項ありの閾値の更新
   end
%%%　その他の結合係数は入力層に向かって順次修正を行う。誤差の計算に注意！
%%%　ステップ２：結合係数(w3)を修正する
  for i=1:N4
	sigma = 0 ;%これが誤差となる。
	for k=1:N5
		sigma = sigma + delta5(k)*w4(i,k) ;%ここが誤差の計算となっている。
    end
    if ACTIVATION == 0
        %シグモイド関数の場合
        delta4(i) = (1/(1+exp(-s4(i))))*(1-1/(1+exp(-s4(i))))*sigma ;%出力層以外のデルタの計算
    else     
        %ReLUの場合
        if s4(i) > 0
            delta4(i)=sigma;
        else
            delta4(i) = ReLU_GAIN*sigma;
        end
    end
	for j = 1:N3
		w3(j,i) = w3(j,i) - eta * delta4(i)*x3(j) + eta_myu * (w3(j,i) - w3_tmp(j,i)) ;
	end
    th4(i) = th4(i) - beta * delta4(i)*x3(j)+ beta_myu * (th4(i) - th4_tmp(i));%ユニットの閾値の更新
  end   
%%%　その他の結合係数は入力層に向かって順次修正を行う。誤差の計算に注意！
%%%　ステップ３：結合係数(w2)を修正する
  for i=1:N3
	sigma = 0 ;%これが誤差となる。
	for k=1:N4
		sigma = sigma + delta4(k)*w3(i,k) ;%ここが誤差の計算となっている。
    end
    if ACTIVATION == 0
        %シグモイド関数の場合
        delta3(i) = (1/(1+exp(-s3(i))))*(1-1/(1+exp(-s3(i))))*sigma ;%出力層以外のデルタの計算
    else
        %ReLUの場合
        if s3(i) > 0
            delta3(i)=sigma;
        else
            delta3(i) = ReLU_GAIN*sigma;
        end
    end
	for j = 1:N2
		w2(j,i) = w2(j,i) - eta * delta3(i)*x2(j) + eta_myu * (w2(j,i) - w2_tmp(j,i)) ;
	end
    th3(i) = th3(i) - beta * delta3(i)*x2(j)+ beta_myu * (th3(i) - th3_tmp(i));%ユニットの閾値の更新
  end
  %%%　ステップ４：結合係数(w1)を修正する
  for i=1:N2
	sigma = 0 ;%これが誤差となる。まずは初期化する。
	for k=1:N3
		sigma = sigma + delta3(k)*w2(i,k) ;%出力層以外の誤差量の計算
    end
    if ACTIVATION == 0
        %シグモイド関数の場合
        delta2(i) = (1/(1+exp(-s2(i))))*(1-1/(1+exp(-s2(i))))*sigma ;
    else
        %ReLUの場合
        if s2(i) > 0
            delta2(i)=sigma;
        else
            delta2(i) = ReLU_GAIN*sigma;
        end
    end
	for j = 1:N1
		w1(j,i) = w1(j,i) - eta * delta2(i)*x1(j) + eta_myu * (w1(j,i) - w1_tmp(j,i)) ;
	end
    th2(i) = th2(i) - beta * delta2(i)*x1(j)+ beta_myu * (th2(i) - th2_tmp(i));%閾値の更新
  end
end
