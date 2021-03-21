import sys, os

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import cv2
import numpy as np

from ImageProcessing.GrayScale import AutoGrayScale


def GlobalThreshold(
    img: np.ndarray,
    threshold: int = 127,
    Type: str = "cv2",
) -> np.ndarray:
    """
    画素値が閾値より大きければある値(白色'255')を割り当て，そうでなければ別の値(黒色)を割り当てる。
    入力が None なら None を、変換に成功すれば閾値処理された2値画像を返す。

    Args:
        img (np.ndarray):
            変換前の画像
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
    """
    if type(img) is not np.ndarray:  # 入力データがndarray型でない場合
        raise ValueError("入力型が異なります。")
    elif len(img.shape) != 2: # 入力データがグレースケール画像でない場合
        raise ValueError("入力はグレースケール画像でなければなりません。")

    try:
        if Type == "cv2":
            _, dst = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
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


def _OtsuThreshold(
    img: np.ndarray,
    min_value: int = 0,
    max_value: int = 255
):
    """
    入力画像が bimodal image (ヒストグラムが双峰性を持つような画像)であることを仮定すると、
    そのような画像に対して、二つのピークの間の値を閾値として選べば良いと考えることであろう。これが大津の二値化の手法である。
    双峰性を持たないヒストグラムを持つ画像に対しては良い結果が得られないことになる。

    Args:
        img (np.ndarray):
            変換前の画像
        min_value (int optional):
            二値化後に置き換えたピクセルの最小値
            default: 0
        max_value (int optional):
            二値化後に置き換えたピクセルの最大値
            default: 255

    Returns:
        img (np.ndarray):
            変換後の画像データ
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


def AdaptiveThreshold(
    img: np.ndarray,
    method=cv2.ADAPTIVE_THRESH_MEAN_C,
    Type=cv2.THRESH_BINARY,
    block_size: int=11,
    C: int=2):
    """
    適応的閾値処理では，画像の小領域ごとに閾値の値を計算する．
    そのため領域によって光源環境が変わるような画像に対しては，単純な閾値処理より良い結果が得られる．
    img is NoneならNoneを、変換に成功すれば閾値処理された2値画像を返す。

    Args:
        img (np.ndarray):
            変換前の画像データ
        method (optional):
            小領域中での閾値の計算方法
            * cv2.ADAPTIVE_THRESH_MEAN_C : 近傍領域の中央値を閾値とする。
            * cv2.ADAPTIVE_THRESH_GAUSSIAN_C : 近傍領域の重み付け平均値を閾値とする。
                                            重みの値はGaussian分布になるように計算。
        Type (optional):
            閾値の処理方法
            * cv2.THRESH_BINARY
            * cv2.THRESH_BINARY_INV
        block_size (int optional):
            閾値計算に使用する近傍領域のサイズ。
            ただし1より大きい奇数でなければならない。
        C (int optional):
            計算された閾値から引く定数。
    """
    if type(img) is not np.ndarray:  # 入力データがndarray型でない場合
        raise ValueError("入力型が異なります。")
    elif len(img.shape) != 2: # 入力データがグレースケール画像でない場合
        raise ValueError("入力はグレースケール画像でなければなりません。")

    img = cv2.adaptiveThreshold(img, 255, method, Type, block_size, C)
    return img


def TwoThreshold(
    img: np.ndarray,
    LowerThreshold: int=0,
    UpperThreshold: int=128,
    PickupColor: int=4,
    Type=cv2.THRESH_BINARY):
    """
    上側と下側の2つの閾値で2値化を行う。
    二値化には大局的閾値処理を用いる。

    Args:
        img (np.ndarray):
            変換前の画像データ
        LowerThreshold (int optional):
            下側の閾値(0 <= th <= 127)
        UpperThreshold (int optional):
            上側の閾値(128 <= th <= 255)
        PickupColor (int optional):
            抽出したい色を指定する。
            * 0: 赤, 1: 緑, 2: 青, 3: 白, 4: 黒色
            default: 4
        Type (optional):
            閾値の処理方法
            ・cv2.THRESH_BINARY
            ・cv2.THRESH_BINARY_INV
            ・cv2.THRESH_TRUNC
            ・cv2.THRESH_TOZERO
            ・cv2.THRESH_TOZERO_INV

    Return:
        IMAGE_bw (np.ndarray):
            変換後の画像データ
    """

    if type(img) is not np.ndarray:  # 入力データがndarray型でない場合
        raise ValueError("入力型が異なります。")
    elif len(img.shape) != 3: # 入力データがグレースケール画像でない場合
        raise ValueError("入力はRGB画像でなければなりません。")

    r, g, b = cv2.split(img)

    # for Red
    _, IMAGE_R_bw = GlobalThreshold(r, LowerThreshold, Type)
    _, IMAGE_R__ = GlobalThreshold(r, UpperThreshold, Type)
    IMAGE_R__ = cv2.bitwise_not(IMAGE_R__)
    # for Green
    _, IMAGE_G_bw = GlobalThreshold(g, LowerThreshold, Type)
    _, IMAGE_G__ = GlobalThreshold(g, UpperThreshold, Type)
    IMAGE_G__ = cv2.bitwise_not(IMAGE_G__)
    # for Blue
    _, IMAGE_B_bw = GlobalThreshold(b, LowerThreshold, Type)
    _, IMAGE_B__ = GlobalThreshold(b, UpperThreshold, Type)
    IMAGE_B__ = cv2.bitwise_not(IMAGE_B__)

    if PickupColor == 0:
        IMAGE_bw = IMAGE_R_bw*IMAGE_G__*IMAGE_B__   # 画素毎の積を計算　⇒　赤色部分の抽出
    elif PickupColor == 1:
        IMAGE_bw = IMAGE_G_bw*IMAGE_B__*IMAGE_R__   # 画素毎の積を計算　⇒　緑色部分の抽出
    elif PickupColor == 2:
        IMAGE_bw = IMAGE_B_bw*IMAGE_R__*IMAGE_G__   # 画素毎の積を計算　⇒　青色部分の抽出
    elif PickupColor == 3:
        IMAGE_bw = IMAGE_R_bw*IMAGE_G_bw*IMAGE_B_bw  # 画素毎の積を計算　⇒　白色部分の抽出
    elif PickupColor == 4:
        IMAGE_bw = IMAGE_R__*IMAGE_G__*IMAGE_B__    # 画素毎の積を計算　⇒　黒色部分の抽出
    else:
        return 5, None

    return 0, IMAGE_bw




if __name__ == "__main__":
    from PIL import Image
    from matplotlib import pyplot as plt
    from src.config.config import cfg

    img_path = cfg.TEST_IMG_ORG_DIR + os.sep + "lena.png"
    # img_path = cfg.TEST_IMG_ORG_DIR + os.sep + "SaltAndPepper.png"
    img = np.array(Image.open(img_path))

    save_dir = os.path.join(cfg.BINARY_IMG_DIR, "two_threshold")

    LowerThreshold=0
    UpperThreshold=128
    Type="cv2"
    # 2つの閾値を用いた二値化
    r, g, b = cv2.split(img)
    # for Red
    IMAGE_R_bw = GlobalThreshold(r, LowerThreshold, Type)
    IMAGE_R_bw_save = Image.fromarray(IMAGE_R_bw)
    IMAGE_R_bw_save.save(save_dir + os.sep + "img_r_low" + ".png")
    IMAGE_R__ = GlobalThreshold(r, UpperThreshold, Type)
    IMAGE_R___save = Image.fromarray(IMAGE_R__)
    IMAGE_R___save.save(save_dir + os.sep + "img_r_up" + ".png")
    IMAGE_R__ = cv2.bitwise_not(IMAGE_R__)
    IMAGE_R__wb_save = Image.fromarray(IMAGE_R__)
    IMAGE_R__wb_save.save(save_dir + os.sep + "img_r_up_bitwise" + ".png")
    # for Green
    IMAGE_G_bw = GlobalThreshold(g, LowerThreshold, Type)
    IMAGE_G_bw_save = Image.fromarray(IMAGE_G_bw)
    IMAGE_G_bw_save.save(save_dir + os.sep + "img_g_low" + ".png")
    IMAGE_G__ = GlobalThreshold(g, UpperThreshold, Type)
    IMAGE_G___save = Image.fromarray(IMAGE_G__)
    IMAGE_G___save.save(save_dir + os.sep + "img_g_up" + ".png")
    IMAGE_G__ = cv2.bitwise_not(IMAGE_G__)
    IMAGE_G__wb_save = Image.fromarray(IMAGE_G__)
    IMAGE_G__wb_save.save(save_dir + os.sep + "img_g_up_bitwise" + ".png")
    # for Blue
    IMAGE_B_bw = GlobalThreshold(b, LowerThreshold, Type)
    IMAGE_B_bw_save = Image.fromarray(IMAGE_B_bw)
    IMAGE_B_bw_save.save(save_dir + os.sep + "img_b_low" + ".png")
    IMAGE_B__ = GlobalThreshold(b, UpperThreshold, Type)
    IMAGE_B___save = Image.fromarray(IMAGE_B__)
    IMAGE_B___save.save(save_dir + os.sep + "img_b_up" + ".png")
    IMAGE_B__ = cv2.bitwise_not(IMAGE_B__)
    IMAGE_B__wb_save = Image.fromarray(IMAGE_B__)
    IMAGE_B__wb_save.save(save_dir + os.sep + "img_b_up_bitwise" + ".png")

    # 閾値を用いて大域的二値化を行う
    #img = GlobalThreshold(img)  # 大域的二値化処理
    # img = GlobalThreshold(img, Type="Otsu")
    # img = GlobalThreshold(img, clearly=True, Type="Otsu")
    # img = GlobalThreshold(img, Type=1) # エラー
    #plt.imshow(img, cmap="gray")
    #plt.show()

