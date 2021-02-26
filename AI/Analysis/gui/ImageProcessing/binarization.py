import sys, os

sys.path.append(".")
sys.path.append("..")

import cv2
import numpy as np

from GrayScale import GrayScale


def GlobalThreshold(
    img: np.ndarray,
    gaussian: bool = False,
    threshold: int = 127,
    Type=cv2.THRESH_BINARY,
) -> np.ndarray:
    """
    画素値が閾値より大きければある値(白色'255')を割り当て，そうでなければ別の値(黒色)を割り当てる。
    入力が None なら None を、変換に成功すれば閾値処理された2値画像を返す。

    Args:
        img (np.ndarray):
            変換前の画像
        gaussian (bool):
            ガウシアンフィルタを適用し，ノイズを除去する．
            True: 適用する
            False: 適用しない
    threshold (int)
        2値化するときの閾値
        0 <= th <= 256
    Type
        閾値の処理方法
        ・cv2.THRESH_BINARY
        ・cv2.THRESH_BINARY_INV
        ・cv2.THRESH_TRUNC
        ・cv2.THRESH_TOZERO
        ・cv2.THRESH_TOZERO_INV

    Returns:
        dst (np.ndarray):
            変換後の画像データ
        response (int):
            1: 変換できました。
            -1: 画像の元データが存在しません。
    """
    if type(img) is not np.ndarray:  # 画像の元データが存在しない場合
        return -1, None

    # RGB画像を入力とする場合
    if len(img.shape) == 3:
        _, img = GrayScale(img)  # RGB -> Gray
    # ガウスフィルタをかけてノイズを除去する
    if gaussian:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    ret, dst = cv2.threshold(img, threshold, 255, Type)
    dst = np.array(dst)  # ndarray型に変換
    return 1, dst


def OtsuThreshold(
    img: np.ndarray, gaussian: bool = False, min_value: int = 0, max_value: int = 255
):
    """
    入力画像が bimodal image (ヒストグラムが双峰性を持つような画像)であることを仮定すると、
    そのような画像に対して、二つのピークの間の値を閾値として選べば良いと考えることであろう。これが大津の二値化の手法である。
    双峰性を持たないヒストグラムを持つ画像に対しては良い結果が得られないことになる。

    Args:
        img (np.ndarray):
            変換前の画像
        gaussian (bool):
            ガウシアンフィルタを適用し，ノイズを除去する．
            True: 適用する
            False: 適用しない
        min_value (int):
            二値化後に置き換えたピクセルの最小値
        max_value (int):
            二値化後に置き換えたピクセルの最大値

    Returns:
        img (np.ndarray):
            変換後の画像データ
        response (int):
            1: 変換できました。
            -1: 画像の元データが存在しません。
    """
    if img is None:  # 画像の元データが存在しない場合
        return -1, None
    # RGB画像を入力とする場合
    if len(img.shape) == 3:
        _, img = GrayScale(img)  # RGB -> Gray
    # ガウスフィルタをかけてノイズを除去する
    if gaussian:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    # ヒストグラムの算出
    hist = [np.sum(img == i) for i in range(256)]
    s_max = (0, -10)

    for th in range(256):
        # クラス1とクラス2の画素数を計算
        n1 = float(sum(hist[:th]))
        n2 = float(sum(hist[th:]))

        # クラス1とクラス2の画素値の平均を計算
        if n1 == 0:
            mu1 = 0
        else:
            mu1 = sum([i * hist[i] for i in range(0, th)]) / n1
        if n2 == 0:
            mu2 = 0
        else:
            mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2

        # クラス間分散の分子を計算
        s = n1 * n2 * (float(mu1) - float(mu2)) ** 2

        # クラス間分散の分子が最大のとき、クラス間分散の分子と閾値を記録
        if s > s_max[1]:
            s_max = (th, s)

    # クラス間分散が最大のとき閾値を取得
    t = s_max[0]

    # 算出した閾値で二値化処理
    img[img < t] = min_value
    img[img >= t] = max_value

    return 1, img


if __name__ == "__main__":
    from DobotFunction.Camera import WebCam_OnOff, Snapshot, Preview

    _, cam = WebCam_OnOff(0)
    _, img = Snapshot(cam)

    # 閾値を用いて大域的二値化を行う
    # _, img = GlobalThreshold(img) # 大域的二値化処理
    # _, img = GlobalThreshold(img, gaussian=True)  # ノイズ除去->二値化処理
    # _, img = OtsuThreshold(img)  # 大津の二値化
    _, img = OtsuThreshold(img, gaussian=True)  # ノイズ除去->大津の二値化
    Preview(img, preview="plt")

