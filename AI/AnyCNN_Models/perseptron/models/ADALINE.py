import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np


class AdalineGD(object):
    """ADAptive Liner NEuronの分類器

    Params
    ------
    eta(float):
        学習率（ より大きく  以下の値)
    n_iter(int):
        トレーニングデータのトレーニング回数
    random_state(int):
        重みを初期化するための乱数シード
    
    属性
    ---
    w_ (1次元配列):
        適合後の重み
    cost_(list):
        各エポックでの誤差平方和コスト関数
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """トレーニングデータに適合させる

        Params
        ------
        X (配列のようなデータ構造) shape = [n_samples, n_features]:
            トレーニングデータ
            n_sampleはサンプルの個数、n_featuresは特徴量の個数

        y (配列のようなデータ構造) shape = [n_samples]:
            目的変数

        Return
        ------
        self (Object):
        """

        # 重みとして使用する乱数を発生させる
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):  # トレーニング回数分だけトレーニングデータを反復
            net_input = self.net_input(X)
            # activate メソッドは単なる恒等関数であるため、
            # このコードでは何の効果もないことに注意。代わりに、
            # 直接 `output = self.net_input(X)` と記述することもできた。
            # activationメソッドの目的は、より概念的なものである。
            # つまり (後ほど説明する) ロジスティクス回帰の分類器を実装するために
            # シグモイド関数に変更することもできる。
            output = self.activation(net_input)
            # 誤差の計算
            errors = y - output
            # 重みの更新
            self.w_[1:] += self.eta * X.T.dot(errors)
            # バイアスの更新
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算
            cost = (errors ** 2).sum() / 2.0
            # コストの格納
            self.cost_.append(cost)

        return self

    def activation(self, X):
        """線形活性化関数の出力を計算"""

        return X

    def net_input(self, X):
        """総入力を計算
        
        Parameter
        ---------
        X (配列のようなデータ構造) shape = [n_samples, n_features]:
            トレーニングデータ
        
        Return
        ------
        (ndarray):
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """1ステップ後のクラスラベルを返す

        Parameters
        ----------
        X (配列のようなデータ構造) shape = [n_samples, n_features]:
            トレーニングデータ

        Return
        ------
        (ndarray):
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class AdalineSGD(object):
    """ADAptive Liner NEuronの分類器

    Params
    ------
    eta(float):
        学習率（ より大きく  以下の値)
    n_iter(int):
        トレーニングデータのトレーニング回数
    shuffle(bool):
        * True: 循環を回避するためにエポックごとにトレーニングデータをシャッフル
    random_state(int):
        重みを初期化するための乱数シード
    
    属性
    ---
    w_ (1次元配列):
        適合後の重み
    cost_(list):
        各エポックですべてのトレーニングサンプルの平均を求める誤差平方和コスト関数
    """
