import sys, os

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import cv2
import numpy as np

from ImageProcessing.GrayScale import AutoGrayScale


def ExtractContours(
    org_img,
    kernelShape=cv2.MORPH_RECT,
    RetrievalMode=cv2.RETR_LIST,
    ApproximateMode=cv2.CHAIN_APPROX_SIMPLE,
    min_area=100,
):
    """
    画像に含まれるオブジェクトの輪郭を抽出する関数。
    黒い背景（暗い色）から白い物体（明るい色）の輪郭を検出すると仮定。

    Args:
        org_img (np.ndarray):
            変換前の画像データ(二値)
        kernelShape
            モルフォロジー変換で使用する入力画像と処理の性質を決める構造的要素
            カーネルの種類
            ・cv2.MORPH_RECT: 矩形カーネル
            ・cv2.MORPH_ELLIPSE: 楕円形カーネル
            ・cv2.MORPH_CROSS: 十字型カーネル
        RetrievalMode
            輪郭の階層情報
            cv2.RETR_LIST: 輪郭の親子関係を無視する。
            親子関係が同等に扱われるので、単なる輪郭として解釈される。
            cv2.RETR_EXTERNAL: 最も外側の輪郭だけを検出するモード
            cv2.RETR_CCOMP: 2レベルの階層に分類する。
                物体の外側の輪郭を階層1、物体内側の穴などの輪郭を階層2として分類。
            cv2.RETR_TREE:  全階層情報を保持する。
        ApproximateMode
            輪郭の近似方法
            cv2.CHAIN_APPROX_NONE: 中間点も保持する。
            cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない。
        min_area : int
            領域が占める面積の閾値を指定

    Returns:

    """
    # 画像が入力されている場合
    if type(org_img) is np.ndarray:
        img_bin = org_img.copy()
        # 画像のチャンネル数が2より大きい場合 ⇒ グレー画像に変換
        img_bin = AutoGrayScale(img_bin)

        """
        モルフォロジー変換
        主に二値画像を対象とし、画像上に写っている図形に対して作用するシンプルな処理．
        クロージング処理
        クロージング処理はオープニング処理の逆の処理を指し、膨張の後に収縮 をする処理．
        前景領域中の小さな(黒い)穴を埋めるのに役立ちます。
        """
        # フィルタの設定
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        # クロージング処理
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

        # 輪郭検出（Detection contours）
        # tmp_img, contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 輪郭近似（Contour approximation）
        # approx = approx_contour(contours)

        # 等高線の描画（Contour line drawing）
        # cp_org_img_for_draw = np.copy(org_img)
        # drawing_edge(cp_org_img_for_draw, approx, min_area)

        # return contours, cp_org_img_for_draw
        # return contours, tmp_img
        return img_bin
    else:
        return None, None


def approx_contour(contours):
    """
    輪郭線の直線近似を行う関数

    Arg:
        contours : OpenCV型
            画像から抽出した輪郭情報

    Return:
        approx (list):
            近似した輪郭情報
    """
    approx = []
    for i in range(len(contours)):
        cnt = contours[i]
        epsilon = 0.001 * cv2.arcLength(cnt, True)  # 実際の輪郭と近似輪郭の最大距離を表し、近似の精度を表すパラメータ
        approx.append(cv2.approxPolyDP(cnt, epsilon, True))
    return approx


def CenterOfGravity(org_img: np.ndarray, contours, cal_Method=0):
    """
    オブジェクトの図心を計算する関数

    Args:
        org_img (np.ndarray):
            図心計算用の画像
    contours : OpenCV型
        画像から抽出した輪郭情報
    cal_Method (int)
        重心計算を行う方法を選択する
        0: 画像から重心を計算
        1: オブジェクトの輪郭から重心を計算

    Return:
        cx, cy (int):
            オブジェクトの重心座標
    """
    # 計算方法が不正の場合
    if cal_Method > 1:
        cal_Method = 1

    img = org_img.copy()

    # -------- #
    # 二値化処理 #
    # -------- #

    # 画像のチャンネル数が2より大きい場合 ⇒ グレー画像に変換
    if len(img.shape) > 2:
        _, img = GrayScale(img)

    # 画像をもとに重心を求める場合
    if cal_Method == 0:
        M = cv2.moments(img_bin, False)

    # 輪郭から重心を求める場合
    else:
        maxCont = contours[0]
        for c in contours:
            if len(maxCont) < len(c):
                maxCont = c

        M = cv2.moments(maxCont)
    if int(M["m00"]) == 0:
        return None

    try:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    except ZeroDivisionError:
        return None

    return (cx, cy)


if __name__ == "__main__":
    from PIL import Image
    from matplotlib import pyplot as plt
    from src.config.config import cfg

    img_path = cfg.TEST_DIR + os.sep + "lena.png"
    img = np.array(Image.open(img_path))

    # ----- #
    # テスト #
    # ----  #
    dst = ExtractContours(img)

    plt.imshow(dst)
    plt.show()
