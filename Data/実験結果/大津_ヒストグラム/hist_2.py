import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot


def class_split(brightness_list, high_mean, low_mean):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    hist = ax.hist(brightness_list, bins=255)

    max_num = np.max(hist[0])

    ax.set_xlim([0, 255])
    ax.set_ylim([0, 1.1 * max_num])

    #ax.set_yticklabels([])

    between = (high_mean + low_mean) / 2

    # しきい値に関しての描画
    #ax.vlines(x=between, ymin=0, ymax=2000, colors="gray", linestyles="--")
    #ax.text(x=0.98 * between, y=max_num, s="しきい値(t)", ha="right")

    # クラスに関しての描画
    #ax.text(x=0.7 * between, y=max_num * 0.8, s="クラス0", fontsize=20, ha="center")
    #ax.text(x=1.3 * between, y=max_num * 0.8, s="クラス1", fontsize=20, ha="center")

    # グラフ右と上の軸を消す
    pyplot.gca().spines['right'].set_visible(False)
    pyplot.gca().spines['top'].set_visible(False)

    pyplot.savefig('class_split.png')


def var_between_classes(brightness_list, high_mean, low_mean):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    hist = ax.hist(brightness_list, bins=255)

    max_num = np.max(hist[0])

    ax.set_xlim([0, 255])
    ax.set_ylim([0, 1.1 * max_num])

    ax.set_yticklabels([])

    between = (high_mean + low_mean) / 2

    # しきい値に関しての描画
    ax.vlines(x=between, ymin=0, ymax=500, colors="gray", linestyles="--")

    # 各クラスの平均に関する描画
    ax.vlines(x=high_mean, ymin=0, ymax=500, colors="C1", linestyles="--")
    #ax.text(s="クラス1の平均", x=high_mean, y=1.02 * max_num, color="C1", ha="right")
    ax.vlines(x=low_mean, ymin=0, ymax=500, colors="C1", linestyles="--")
    #ax.text(s="クラス0の平均", x=low_mean, y=1.02 * max_num, color="C1", ha="left")

    # クラス間分散に関する描画
    ax.text(s="2つのクラスの離れ具合\n(クラス間分散)", x=between,
            y=0.88 * max_num, ha="center", va="top")
    ax.annotate('',
                xy=(high_mean, 0.9 * max_num), xycoords='data',
                xytext=(low_mean, 0.9 * max_num), textcoords='data', fontsize=0,
                arrowprops=dict(shrink=0, width=2, headwidth=8, headlength=10,
                                connectionstyle='arc3', facecolor='C1', edgecolor='C1'))
    ax.annotate('',
                xy=(low_mean, 0.9 * max_num), xycoords='data',
                xytext=(high_mean, 0.9 * max_num), textcoords='data', fontsize=0,
                arrowprops=dict(shrink=0, width=2, headwidth=8, headlength=10,
                                connectionstyle='arc3', facecolor='C1', edgecolor='C1'))

    pyplot.savefig('var_between_classes.png')


def var_in_each_classes(brightness_list, high_mean, low_mean):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    hist = ax.hist(brightness_list, bins=255)

    max_num = np.max(hist[0])

    ax.set_xlim([0, 255])
    ax.set_ylim([0, 1.1 * max_num])

    ax.set_yticklabels([])

    between = (high_mean + low_mean) / 2

    # しきい値に関しての描画
    ax.vlines(x=between, ymin=0, ymax=500, colors="gray", linestyles="--")

    # 各クラスの平均に関する描画
    ax.vlines(x=high_mean, ymin=0, ymax=500, colors="C1", linestyles="--")
    #ax.text(s="クラス1の平均", x=high_mean, y=1.02 * max_num, color="C1", ha="right")
    ax.vlines(x=low_mean, ymin=0, ymax=500, colors="C1", linestyles="--")
    #ax.text(s="クラス0の平均", x=low_mean, y=1.02 * max_num, color="C1", ha="left")

    # クラス0内の分散に関する描画
    ax.text(s="クラス0のばらつき\n(クラス内分散0)", x=low_mean,
            y=0.18 * max_num, ha="center", va="top")
    ax.annotate('',
                xy=(0, 0.2 * max_num), xycoords='data',
                xytext=(2 * low_mean, 0.2 * max_num), textcoords='data', fontsize=0,
                arrowprops=dict(shrink=0, width=2, headwidth=8, headlength=10,
                                connectionstyle='arc3', facecolor='C1', edgecolor='C1'))
    ax.annotate('',
                xy=(2 * low_mean, 0.2 * max_num), xycoords='data',
                xytext=(0, 0.2 * max_num), textcoords='data', fontsize=0,
                arrowprops=dict(shrink=0, width=2, headwidth=8, headlength=10,
                                connectionstyle='arc3', facecolor='C1', edgecolor='C1'))

    # クラス1内の分散に関する描画
    ax.text(s="クラス1のばらつき\n(クラス内分散1)", x=high_mean,
            y=0.18 * max_num, ha="center", va="top")
    ax.annotate('',
                xy=(255, 0.2 * max_num), xycoords='data',
                xytext=(255 - (255 - high_mean) * 2, 0.2 * max_num), textcoords='data', fontsize=0,
                arrowprops=dict(shrink=0, width=2, headwidth=8, headlength=10,
                                connectionstyle='arc3', facecolor='C1', edgecolor='C1'))
    ax.annotate('',
                xy=(255 - (255 - high_mean) * 2, 0.2 * max_num), xycoords='data',
                xytext=(255, 0.2 * max_num), textcoords='data', fontsize=0,
                arrowprops=dict(shrink=0, width=2, headwidth=8, headlength=10,
                                connectionstyle='arc3', facecolor='C1', edgecolor='C1'))

    pyplot.savefig('var_in_each_classes.png')


def main():
    # グラフの体裁を整える
    pyplot.rcParams['xtick.direction'] = 'in' # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    pyplot.rcParams['ytick.direction'] = 'in' # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    pyplot.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
    pyplot.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
    pyplot.rcParams['font.size'] = 11  # フォントの大きさ
    pyplot.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ

    # グラフ右と上の軸を消す
    pyplot.gca().spines['right'].set_visible(False)
    pyplot.gca().spines['top'].set_visible(False)

    high_mean = 220
    low_mean = 40
    brightness_list = np.random.normal(high_mean, 25, 4000)
    brightness_list = np.append(
        brightness_list, np.random.normal(low_mean, 35, 2600))

    # クラス分割の図
    #font = {"family": "IPAexGothic"}
    #mpl.rc('font', **font)

    class_split(brightness_list, high_mean, low_mean)
    var_between_classes(brightness_list, high_mean, low_mean)
    var_in_each_classes(brightness_list, high_mean, low_mean)


if __name__ == "__main__":
    main()
