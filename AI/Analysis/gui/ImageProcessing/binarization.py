import sys, os

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import cv2
import numpy as np

from ImageProcessing.GrayScale import AutoGrayScale


def GlobalThreshold(
    img: np.ndarray, clearly: bool = False, threshold: int = 127, Type: str = "cv2",
) -> np.ndarray:
    """
    画素値が閾値より大きければある値(白色'255')を割り当て，そうでなければ別の値(黒色)を割り当てる。
    入力が None なら None を、変換に成功すれば閾値処理された2値画像を返す。

    Args:
        img (np.ndarray):
            変換前の画像
        clearly (bool optional):
            ガウシアンフィルタを用いて入力画像のノイズを除去する．
            * True: 適用する
            * False: 適用しない: default
        threshold (int optional):
            2値化するときの閾値(0 <= th <= 256)
            default: 127
        Type (str optional):
            閾値の処理方法
            * "cv2": OpenCVの関数を用いて二値化を行う: default
            * "Otsu: 大津の二値化処理

    Returns:
        dst (np.ndarray):
            変換後の画像データ(Errorが発生した場合: None)
        response (int):
            1: 変換できました。
            -1: 画像の元データが存在しません。
    """
    if type(img) is not np.ndarray:  # 画像の元データが存在しない場合
        return -1, None

    # ガウスフィルタをかけてノイズを除去する
    if clearly:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    # RGB画像を入力とした場合は、グレー化
    img = AutoGrayScale(img)  # RGB -> Gray

    try:
        if Type == "cv2":
            ret, dst = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        elif Type == "Otsu":
            dst = _OtsuThreshold(img)
        else:
            raise Exception("選択した処理方法が存在しません．")
    except Exception as e:
        print("大域的二値化処理中のエラー: ", e)
        return None
    else:
        dst = np.array(dst)  # ndarray型に変換
        return dst


def _OtsuThreshold(img: np.ndarray, min_value: int = 0, max_value: int = 255):
    """
    入力画像が bimodal image (ヒストグラムが双峰性を持つような画像)であることを仮定すると、
    そのような画像に対して、二つのピークの間の値を閾値として選べば良いと考えることであろう。これが大津の二値化の手法である。
    双峰性を持たないヒストグラムを持つ画像に対しては良い結果が得られないことになる。

    Args:
        img (np.ndarray):
            変換前の画像
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

    return img


if __name__ == "__main__":
    # from DobotFunction.Camera import WebCam_OnOff, Snapshot, Preview
    # _, cam = WebCam_OnOff(0)
    # _, img = Snapshot(cam)
    # Preview(img, preview="plt")

    from PIL import Image
    from matplotlib import pyplot as plt
    from src.config.config import cfg

    # img_path = cfg.TEST_DIR + os.sep + "lena.png"
    img_path = cfg.TEST_IMG_ORG_DIR + os.sep + "SaltAndPepper.png"
    img = np.array(Image.open(img_path))

    # 閾値を用いて大域的二値化を行う
    img = GlobalThreshold(img)  # 大域的二値化処理
    # img = GlobalThreshold(img, Type="Otsu")
    # img = GlobalThreshold(img, clearly=True, Type="Otsu")
    # img = GlobalThreshold(img, Type=1) # エラー

    plt.imshow(img, cmap="gray")
    plt.show()

