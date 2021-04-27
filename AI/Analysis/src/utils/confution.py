import os
import sys
sys.path.append(".")
sys.path.append("..")
import shutil
from glob import glob

import numpy as np
import openpyxl
import pandas as pd
from matplotlib import pyplot as plt
from openpyxl.drawing.image import Image

from config.config import cfg
from src.utils.utils import makedir

# ファイルパスをリストで取得
parent_dir = cfg.CONFUTION_DIR  # 読み込むcsvファイルの親ディレクトリ
f_pathes = glob(parent_dir+os.sep+'*.csv', recursive=True)  # 読み出すファイルパス

# 計算用ファイルを一時的に保存するtempディレクトリ
temp_dir = makedir(parent_dir, dir_name='temp', newly=True)

# 計算結果を保存する dist ディレクトリ
dist_dir = makedir(parent_dir, dir_name='dist', newly=True)

# index 名
index_name = [
    '0',  '10', '100', '105', '110', '115', '120', '125',
    '130', '135', '140', '145', '15', '150', '155', '160',
    '165', '170', '175', '20', '25', '30', '35', '40',
    '45', '5', '50', '55', '60', '65', '70', '75',
    '80', '85', '90', '95'
]


def _Calc_Acc(df: pd.DataFrame, tp: list, n: int = 3) -> float:
    """クラス数に対する正解率の平均値を計算する関数

    Args:
        df (pd.DataFrame): 混合行列のデータ
        tp (list): クラスごとの真陽性枚数の計算結果のリスト
        n (int, optional): 浮動小数点数の丸め位置. Defaults to 3.

    Return:
        acc (float): 正解率の平均値
    """
    all = sum(df.sum(axis='index')) # すべての要素の合計数を計算

    acc = round(sum(tp)/all, n)
    return acc


def _Calc_TP(df: pd.DataFrame) -> list:
    """真陽性枚数の計算

    Arg:
        df (pd.DataFrame): 混合行列のデータ

    Return:
        tp (list): クラスごとの計算結果のリスト
    """
    tp = []

    for i in range(len(df)):
        tp.append(df.iat[i, i])  # True Positive
    return tp


def _Calc_FP(df: pd.DataFrame, tp: list) -> list:
    """偽陽性枚数の計算

    Args:
        df (pd.DataFrame): 混合行列のデータ
        tp (list): クラスごとの真陽性枚数の計算結果のリスト

    Return:
        fp (list): クラスごとの計算結果のリスト
    """
    fp = []

    for i in range(len(df)):
        fp.append(sum(df.iloc[i, :]) - tp[i])  # False Positive
    return fp


def _Calc_FN(df: pd.DataFrame, tp: list) -> list:
    """偽陰性枚数の計算

    Args:
        df (pd.DataFrame): 混同行列のデータ
        tp (list): クラスごとの真陽性枚数の計算結果のリスト

    Return:
        fn (list): クラスごとの計算結果のリスト
    """
    fn = []

    for i in range(len(df)):
        fn.append(sum(df[str(i*5)]) - tp[i])  # False Negative
    return fn


def _Calc_TN(df: pd.DataFrame, tp: list, fn: list, fp: list) -> list:
    """真陰性枚数の計算

    Args:
        df (pd.DataFrame): 混同行列のデータ
        tp (list): クラスごとの真陽性枚数の計算結果のリスト
        fp (list): クラスごとの偽陽性枚数の計算結果のリスト
        fn (list): クラスごとの偽陰性枚数の計算結果のリスト

    Return:
        tn (list): クラスごとの計算結果のリスト
    """
    all = sum(df.sum(axis='index')) # すべての要素の合計数を計算
    tn = []

    for i in range(len(df)):
        tn.append(all - tp[i]-fp[i]-fn[i])  # True Negative
    return tn


def _Calc_TPR(tp: list, fn: list, n: int = 3) -> list:
    """真陽性率の計算

    Args:
        tp (list): クラスごとの真陽性枚数の計算結果のリスト
        fn (list): クラスごとの偽陰性枚数の計算結果のリスト
        n (int, optional): 浮動小数点数の丸め位置. Defaults to 3.

    Return:
        tpr (list): クラスごとの計算結果のリスト
    """
    tpr = []

    for i in range(len(tp)):
        tpr.append(round(tp[i]/(tp[i] + fn[i]), n))
    return tpr


def _Calc_FNR(tpr: list, n: int = 3) -> list:
    """偽陰性率の計算

    Args:
        tpr (list): クラスごとの真陽性率の計算結果のリスト
        n (int, optional): 浮動小数点数の丸め位置. Defaults to 3.

    Return:
        fnr (list): クラスごとの計算結果のリスト
    """
    fnr = []

    for i in range(len(tpr)):
        fnr.append(round(1-tpr[i], n))
    return fnr


def _Calc_TNR(tn: list, fp: list, n: int = 3) -> list:
    """真陰性率の計算

    Args:
        tn (list): クラスごとの真陰性枚数の計算結果のリスト
        fp (list): クラスごとの偽陰性枚数の計算結果のリスト
        n (int, optional): 浮動小数点数の丸め位置. Defaults to 3.

    Return:
        tnr (list): クラスごとの計算結果のリスト
    """
    tnr = []

    for i in range(len(tn)):
        tnr.append(round(tn[i]/(tn[i]+fp[i]), n))
    return tnr


def _Calc_FPR(tnr: list, n: int = 3) -> list:
    """偽陽性率の計算

    Args:
        tnr (list): クラスごとの真陰性率の計算結果のリスト
        n (int, optional): 浮動小数点数の丸め位置. Defaults to 3.

    Return:
        fpr (list): クラスごとの計算結果のリスト
    """
    fpr = []

    for i in range(len(tnr)):
        fpr.append(round(1 - tnr[i], n))
    return fpr


def _Calc_TPA(tp: list, fp: list, n: int = 3) -> list:
    """真陽性精度の計算

    Args:
        tp (list): クラスごとの真陽性枚数の計算結果のリスト
        fp (list): クラスごとの偽陽性枚数の計算結果のリスト
        n (int, optional): 浮動小数点数の丸め位置. Defaults to 3.

    Return:
        tpa (list): クラスごとの計算結果のリスト
    """
    tpa = []

    for i in range(len(tp)):
        tpa.append(round(tp[i]/(tp[i]+fp[i]), n))
    return tpa


def _Calc_TNA(tn: list, fn: list, n: int = 3) -> list:
    """真陰性精度の計算

    Args:
        tn (list): クラスごとの真陰性枚数の計算結果のリスト
        fn (list): クラスごとの偽陰性枚数の計算結果のリスト
        n (int, optional): 浮動小数点数の丸め位置. Defaults to 3.

    Returns:
        tna (list): クラスごとの計算結果のリスト
    """
    tna = []

    for i in range(len(tn)):
        tna.append(round(tn[i]/(tn[i]+fn[i]), n))
    return tna


def _Drowing_Table(df: pd.DataFrame, save_path: str = "./table.png", fig_size: tuple= (20, 20)):
    """混同行列の表をカラー付き画像で保存する関数

    Args:
        df (pd.DataFrame): 混同行列のデータ
        save_path (str, optional): 拡張子付きの表の保存先. default to "./table.png".
        fig_size (tuple, optional): 保存する表の大きさ
    """
    # fig の準備
    data = df.values
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    # dataをMin-Max 正規化
    norm_data = (data- np.min(data)) / (np.max(data) - np.min(data))
    # cmapを使ってデータのカラー配列を作る
    cm = plt.get_cmap('coolwarm')
    color = np.full_like(data, 0, dtype=object)
    color = cm(norm_data)

    # 表の定義
    ax.axis('off')
    the_table = ax.table(
        cellText=data,
        colLabels = df.columns.values.astype(str),
        rowLabels = df.index.values.astype(str),
        loc = "center",
        cellColours=color
    )

    # 表をfigure全体に表示させる
    for pos, cell in the_table.get_celld().items():
        cell.set_height(1/len(data))

    # 表を保存
    plt.savefig(save_path)


def main():
    for f_path in f_pathes:
        print(f_path)
        f_name, ext = os.path.splitext(os.path.basename(f_path))  #パスから filename と 拡張子を抽出
        save_path = os.path.join(temp_dir, f_name + '_T'+ ext) # 保存先のファイルパス
        save_sort_path = os.path.join(temp_dir, f_name + '_sort' + ext)
        # 表を画像として保存するパスを設定
        img_path = os.path.join(temp_dir, f_name + '.png')
        # Excel ファイルの保存先のパス
        save_excel_path = os.path.join(dist_dir, f_name + '_sort' + '.xlsx')

        df = pd.read_csv(f_path, header=None, names=index_name)  # csv ファイルを読み出し
        df.index = index_name # 列名を追加
        df.T  # 配列を転置する
        df.to_csv(save_path)  # 転置したデータを一度保存

        df = pd.read_csv(save_path, header=None)  # 保存したデータを再度読み出し
        # 1列目の値をもとに行の順番をソート
        df_row_sort = df.sort_values(0, axis=0, ascending=True, na_position='first')
        # 1行目の値をもとに列の順番をソート
        df_col_sort = df_row_sort.sort_values(0, axis=1, ascending=True, na_position='first')


        df_col_sort.to_csv(save_sort_path, header=None, index=False) # header と index を除いて保存

        df = pd.read_csv(save_sort_path, index_col=0)  # 保存したデータを再々度読み出し

        # --------------------- #
        # 4つの変数を計算 #
        # --------------------- #
        tp = _Calc_TP(df)
        fp = _Calc_FP(df, tp)
        fn = _Calc_FN(df, tp)
        tn = _Calc_TN(df, tp=tp, fn=fn, fp=fp)

        # ------------------------------------- #
        # クラスごとの評価指標の計算 #
        # ------------------------------------- #
        tpr = _Calc_TPR(tp, fn)
        fnr = _Calc_FNR(tpr)
        tnr = _Calc_TNR(tn, fp)
        fpr = _Calc_FPR(tnr)
        tpa = _Calc_TPA(tp, fp)
        tna = _Calc_TNR(tn, fp)

        Evaluation_tuple = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "真陽性率(TPR)": tpr,
            "偽陰性率(FNR)": fnr,
            "真陰性率(TNR)": tnr,
            "偽陽性率(FPR)": fpr,
            "真陽性精度(TPA)": tpa,
            "真陰性精度(TNA)": tna,
        }
        res = pd.DataFrame(Evaluation_tuple)

        # -------------------------------- #
        # 評価指標の平均値の計算 #
        # -------------------------------- #
        acc = _Calc_Acc(df, tp)

        Evaluation_tuple = {
            "Acc": [acc],
        }
        res2 = pd.DataFrame(Evaluation_tuple)

        # xslx ファイルに書き出し
        with pd.ExcelWriter(save_excel_path ) as writer:
            df.to_excel(writer, sheet_name="Confusion_matrix")
            res.to_excel(writer, sheet_name="Evaluation_index", index=False)
            res2.to_excel(writer, sheet_name="Average_index", index=False)

        #混同行列の表を作成
        _Drowing_Table(df, save_path=img_path)

        # 混同行列の表を Excel ファイルに保存
        img = Image(img_path) # 画像を読み出し
        try:
            wb = openpyxl.load_workbook(save_excel_path)
            ws = wb.create_sheet(title="Image")
            ws.add_image(img, 'A1')
            wb.save(save_excel_path)
        except Exception as e:
            print('Error: ', e)
        finally:
            wb.close()

if __name__ == '__main__':
    main()
    print('処理が完了しました.')