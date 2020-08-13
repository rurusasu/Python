# cording: utf-8

from collections import defaultdict
import numpy as np
import matplotlib.pylab as plt

class LearningVisualizationCallback:
    """学習曲線を可視化するためのコールバッククラス"""

    def __init__(self, found_score_functions, fig=None, ax=None):
        self._function_histories = defaultdict(list)
        self._function_history_lines = {} # 関数ごとに渡された値を格納するための辞書
        self._found_BestScore_functions = set(found_score_functions)  # ベストスコアを見たい関数を設定する
        self._function_type_fonund_BestScore = defaultdict(bool)    #ベストスコアを見たい関数が入力ラベル内にあるか判定　あればTrue
        self._best_socre_vlines = {} # ベストスコアの縦線の設定を保存するための辞書
        self._best_score_texts = {}  # ベストスコアの文字列の設定を保存するための辞書
        # 描画領域を初期化する
        self._fig = fig # fig:figure(図)
        self._ax = ax   # ax:axes(座標軸)
        if self._fig is None and self._ax is None:
            self._fig, self._ax = plt.subplots()
        self._ax.set_title('learning curve')
        self._ax.set_xlabel('epoch')
        self._ax.set_ylabel('score')
        self._fig.canvas.draw()
        self._ax.show()

    def one_epoch_end(self, data):
        for func, score in data.items():
            # 各関数のスコアを保存する
            self._function_histories[func].append(score)
            # 初回だけの設定
            if len(score) == 1:
                # 関数の種別を保存する
                for found_BestScore_function in self._found_BestScore_functions:
                    if found_BestScore_function in func:
                        self._function_type_fonund_BestScore[func] = True
                        break
                # スコアの履歴を描画するオブジェクトを生成する
                history_line, = self._ax.plot([], []) # X軸, Y軸の値を格納するために空配列を設定
                self._function_history_lines[func] = history_line
                history_line.set_label(func)
                # もし、ラベルに文字列'val'が含まれていない場合
                if 'val' not in func:
                    history_line.set_linestyle('--')
                else:
                    # ベストスコアの線を描画するオブジェクトを生成する
                    best_vline = self._ax.axvline(0)
                    best_vline.set_color(history_line.get_color()) # ベストスコアの線の色を描画する線の色と同じにする
                    best_vline.set_linestyle(':')
                    self._best_socre_vlines[func] = best_vline # それぞれの関数にベストスコアの線の設定をする
                    # ベストスコアの文字列を描画するオブジェクトを生成する
                    vpos = 'top' if self._function_type_fonund_BestScore[func] else 'bottom'
                    best_text = self._ax.text(0, 0, '', 
                                              va=vpos, ha='right', weight='bold')
                    best_text.set_color(history_line.get_color())
                    self._best_score_texts[func] = best_text # それぞれの関数にベストスコアの線の設定をする

        # 描画内容を更新する。
        for func, socre in self._function_histories.items():   
            # グラフデータを更新する
            history_line = self._function_history_lines[func]
            history_line.set_data(np.arange(len(score)), score)
            if 'val' in func:
                if self._function_type_fonund_BestScore[func]:
                    best_score_find_func = np.max
                    best_epoch_find_func = np.argmax
                else:
                    best_score_find_func = np.min
                    best_epoch_find_func = np.argmin
                best_score = best_epoch_find_func(score)
                # 縦線
                best_epoch = best_epoch_find_func(score)
                best_vline = self._best_socre_vlines[func]
                best_vline.set_xdata(best_epoch)
                # テキスト
                best_text = self._best_score_texts[func]
                best_text.set_text(
                    'epoch:{}, score:{:.4f}'.format(best_epoch, best_score))
                best_text.set_x(best_epoch)
                best_text.set_y(best_score)

        # グラフの見栄えを調整する
        self._ax.legend()
        self._ax.relim()
        self._ax.autoscale_view()

        # 再描画する
        plt.pause(0.001)



def PlanePlot(X, Y):
    """
    """
    
    plt.plot(X, Y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def twoD_ArrayQuiver(X, Y, Z):
    """
    Represent a 2D array with arrows
    
    Parameters
    ----------
    X : numpy.ndarray
        メッシュのX座標
    Y : numpy.ndarray
        メッシュのY座標
    Z : numpy.ndarray
        メッシュのZ座標
    """

    plt.figure()
    plt.quiver(X, Y, -Z[0], -Z[1], angles='xy', color='#666666')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()


def two_D_PointPlot(X, Y):
    """
    """
    plt.plot(X, Y, 'o')
    plt.xLabel('x0')
    plt.ylabel('x1')
    plt.show()


if __name__ == "__main__":
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)

    # 平面に計算結果をプロットする
    #Y = 0.01*x0**2 + 0.1*x0
    #PlanePlot(x0, Y)

    X, Y = np.meshgrid(x0, x1)  # meshgrid：配列の要素から格子列を生成する

    X = X.flatten()  # flatten：配列を1次元に変換する
    Y = Y.flatten()

    x = np.array([X, Y])
    Z = np.sum(x**2, axis=0)

    twoD_ArrayQuiver(X, Y, Z)
