import sys

sys.path.append(".")
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def ReadCSV(path):
    try:
        df = pd.read_csv(path, header=0)
    except:
        print("CSVファイルを読み込めませんでした。")
        df = None

    print(df)

    return df


def DataSet_choice(df, standardized=True, plot=True):
    #    df = NoIndex_NoHeader(df)
    # 1-100行目の目的変数を抽出
    y = df.iloc[0:100, 4].values
    # Iris-setosaを-1, Iris-virginicaを1に変換
    y = np.where(y == "setosa", -1, 1)
    # 1-100行目の1, 3列目の抽出
    X = df.iloc[0:100, [0, 2]].values

    if standardized:
        X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    if plot:
        # 品種 setosa のプロット(赤の ○ )
        plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
        # 品種 versicolor のプロット(青の x)
        plt.scatter(
            X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="versicolor"
        )
        # 軸ラベルの設定
        plt.xlabel("sepal length [cm]")
        plt.ylabel("petal length [cm]")
        # 凡例の設定 (左上に配置)
        plt.legend(loc="upper left")
        plt.show()

    return X, y


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # マーカーとカラーマップの準備
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])
    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )

    # 各特徴量を1次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # グラフごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor="black",
        )

    # テストサンプルを目立たせる（点を〇で表示）
    if test idx:
        # すべてのサンプルをプロット
        x_test, y_test = X[test_idx,:], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                        c='',
                        edgecolor='black',
                        alpha=1.0,
                        linewidth=1,
                        marker='o',
                        s=100,
                        label='test set')

if __name__ == "__main__":
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # path = None
    df = ReadCSV(path)
    print(df.tail())

