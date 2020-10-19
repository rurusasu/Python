import os
import sys

sys.path.append(".")
sys.path.append("..")

from lib.configs import cfg, parse_args
from lib.utils import *
from models.perceptron import *
from models.ADALINE import *

import matplotlib.pyplot as plt


def test_plot(net, plot_data, LogPlot=(False, False)):
    LogPlotX = LogPlot[0]
    LogPlotY = LogPlot[1]

    # エポックと誤分類誤差の関係の折れ線グラフをプロット
    # 両方の軸を対数表示する場合
    if LogPlotX is True and LogPlotY is True:
        plt.plot(
            np.log10(range(1, len(plot_data) + 1)), np.log10(plot_data), marker="o"
        )
        plt.xlabel("log(Epochs)")
        plt.ylabel("log(Sum-squared-error)")
    # X軸を対数表示する場合
    elif LogPlotX:
        plt.plot(np.log10(range(1, len(plot_data) + 1)), plot_data, marker="o")
        plt.xlabel("log(Epochs)")
        plt.ylabel("Sum-squared-error")
    # Y軸を対数表示する場合
    elif LogPlotY:
        plt.plot(range(1, len(plot_data) + 1), np.log10(plot_data), marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("log(Sum-squared-error)")
    # 普通にプロットする場合
    else:
        plt.plot(range(1, len(plot_data) + 1), plot_data, marker="o")
        # 軸のラベルの設定
        plt.xlabel("Epochs")
        plt.ylabel("Number of update")

    plt.title("Learning rate {}".format(net.eta))
    # 図の表示
    plt.show()

    # 決定領域のプロット
    plot_decision_regions(X, y, classifier=net)
    # 軸ラベルの設定
    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")
    # 凡例の設定 (左上に配置)
    plt.legend(loc="upper left")
    # 図の表示
    plt.show()


def training(X, y, args):
    eta = args.eta
    n_iter = args.iter

    if args.network == "perseptron":
        net = Perceptron(eta=eta, n_iter=n_iter).fit(X, y)
        PlotData_ = net.errors_
    elif args.network == "adaline":
        if args.optimizer == "GD":
            net = AdalineGD(eta=eta, n_iter=n_iter).fit(X, y)
        elif args.optimizer == "SGD":
            net = AdalineSGD(eta=eta, n_iter=n_iter).fit(X, y)
        else:
            print(
                "NetWork モデル {} には optimizer {} は実装されていません。".format(
                    args.network, args.optimizer
                )
            )
        PlotData_ = net.cost_
    else:
        print("NetWork モデル {} をロードできませんでした。".format(args.network))
        return

    # net.fit(X, y)
    test_plot(net, PlotData_, LogPlot=(False, True))


if __name__ == "__main__":
    args = parse_args()
    # args.datasets = "boston"
    args.network = "adaline"
    args.optimier = "SGD"
    args.eta = 0.01
    #path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    path = os.path.join(cfg.DATASETS_DIR, args.datasets + ".csv")

    df = ReadCSV(path)

    # print(df.index.values)
    # print(df.columns.values)
    X, y = DataSet_choice(df, plot=False)

    training(X, y, args)
    """
    # パーセプトロンのオブジェクトの生成 (インスタンス化)
    ppn = Perceptron(eta=0.1, n_iter=10)
    # トレーニングデータへのモデルの適合
    ppn.fit(X, y)
    # エポックと誤分類誤差の関係の折れ線グラフをプロット
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
    # 軸のラベルの設定
    plt.xlabel("Epochs")
    plt.ylabel("Number of update")
    # 図の表示
    plt.show()

    # 決定領域のプロット
    plot_decision_regions(X, y, classifier=ppn)
    # 軸ラベルの設定
    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")
    # 凡例の設定 (左上に配置)
    plt.legend(loc="upper left")
    # 図の表示
    plt.show()
    """
