import numpy as np
import pandas as pd
import sys

sys.path.append('.')
sys.path.append('..')


def ext_tp(df: pd.DataFrame) -> np.array:
    """TPの要素数を返す関数

    Param
    -----
    df (pandas DataFrame):
        ConfusionMatrixが代入された正方行列

    Return
    ------
    tp (np array):
        TruePositive配列
    """

    matrix = df.values  # DataFrame -> ndarray
    tp = np.diag(matrix)  # 対角成分のみ抽出

    return tp


def ext_fp(df: pd.DataFrame) -> np.array:
    """FPの要素数を返す関数

    Param
    -----
    df (pandas DataFrame):
        ConfusionMatrixが代入された正方行列

    Return
    ------
    fp (np array):
        FalsePositive配列
    """

    tp = ext_tp(df)
    matrix = df.values  # DataFrame -> ndarray

    tp_fp = np.sum(matrix, axis=0)
    fp = tp_fp - tp

    return fp


def ext_fn(df: pd.DataFrame) -> np.array:
    """FPの要素数を返す関数

    Param
    -----
    df (pandas DataFrame):
        ConfusionMatrixが代入された正方行列

    Return
    ------
    fn (np array):
        FalseNegative配列
    """

    tp = ext_tp(df)
    matrix = df.values  # DataFrame -> ndarray

    tp_fn = np.sum(matrix, axis=1)
    fn = tp_fn - tp

    return fn


def summary_of_indicators(df: pd.DataFrame) -> dict:
    """TP FP をまとめた辞書を作成する関数

    Param
    -----
    df (pandas DataFrame):
        ConfusionMatrixが代入された正方行列

    Return
    ------
    ind (dict):
        indicator(指標)をまとめた辞書
    """

    tp = ext_tp(df)
    fp = ext_fp(df)
    fn = ext_fn(df)

    ind = {'tp': tp,
           'fp': fp,
           'fn': fn}

    return ind

#------------------------------------------------------#


def calc_prec(df: pd.DataFrame, indicators: dict) -> dict:
    """micro Precision および macro Precisionを計算する関数

    Params
    ------
    df (pandas DataFrame):
        ConfusionMatrixが代入された正方行列
    indicators (dict):
        ConfusionMatrixから抽出したIndicators(指標)

    Return
    ------
    prec (dict):
        micro Precision と macro Precision を合わせた辞書
    """

    matrix = df.values  # DataFrame -> ndarray
    tp, fp = indicators['tp'], indicators['fp']
    tp_fp = tp + fp

    prec = tp.astype(np.float16) / \
        tp_fp.astype(np.float16)  # 要素ごとのPrecisionの計算

    micro_prec = np.sum(tp, dtype='float16')/np.sum(tp_fp, dtype='float16')
    macro_prec = np.average(prec)

    prec = {'micro_prec': micro_prec,
            'macro_prec': macro_prec}

    return prec


def calc_recall(df: pd.DataFrame, indicators: dict) -> dict:
    """micro Recall および macro Recallを計算する関数

    Params
    ------
    df (pandas DataFrame):
        ConfusionMatrixが代入された正方行列
    indicators (dict):
        ConfusionMatrixから抽出したIndicators(指標)

    Return
    ------
    prec (dict):
        micro Recall と macro Recall を合わせた辞書
    """

    matrix = df.values  # DataFrame -> ndarray
    tp, fn = indicators['tp'], indicators['fn']
    tp_fn = tp + fn

    rec = tp.astype(np.float16) / \
        tp_fn.astype(np.float16)  # 要素ごとのRecallの計算

    micro_rec = np.sum(tp, dtype='float16')/np.sum(tp_fn, dtype='float16')
    macro_rec = np.average(rec)

    rec = {'micro_recall': micro_rec,
           'macro_recall': macro_rec}

    return rec


def calc_f(precision: dict, recall: dict) -> dict:
    """micro F値 および macro F値を計算する関数

    Params
    ------
    precision (dict):
        micro-macro Precision を含む辞書
    recall (duct):
        micro-macro Recall を含む計算

    Returns
    -------
    f (dict):
        micro F値 と macro F値 を合わせた辞書
    """
    micro_prec, macro_prec = precision['micro_prec'], precision['macro_prec']

    micro_rec, macro_rec = recall['micro_recall'], recall['macro_recall']

    micro_f = 2*((micro_rec*micro_prec)/(micro_rec+micro_prec))
    macro_f = 2*((macro_rec*macro_prec)/(macro_rec+macro_prec))

    f = {'micro_f': micro_f.astype(np.float16),
         'macro_f': macro_f.astype(np.float16)}

    return f


def calc_indexes(df: pd.DataFrame) -> pd.DataFrame:
    """precision, recall, F値の計算をまとめて行う関数

    Param
    -----
    df (pandas DataFrame):
        ConfusionMatrixが代入された正方行列

    Return
    ------
    indexes (pandas DataFrame):
        3つの指標を micro-macro で計算した結果
    """

    # 指標を計算する。
    ind = summary_of_indicators(df)
    precision = calc_prec(df, ind)
    recall = calc_recall(df, ind)
    f = calc_f(precision, recall)

    # Pandas DetaFrameにまとめる。
    indexes = pd.DataFrame({'micro': {'precision': precision['micro_prec'],
                                      'recall': recall['micro_recall'],
                                      'f': f['micro_f']},
                            'macro': {'precision': precision['macro_prec'],
                                      'recall': recall['macro_recall'],
                                      'f': f['macro_f']}}
                           )

    return indexes


if __name__ == "__main__":
    import os
    from config.config import cfg

    f_path = cfg.TEST_DIR+os.sep+'Confusion_Matrix.csv'
    df = pd.read_csv(f_path, header=None)

    ind = summary_of_indicators(df)

    precision = calc_prec(df, ind)
    recall = calc_recall(df, ind)

    f = calc_f(precision, recall)
