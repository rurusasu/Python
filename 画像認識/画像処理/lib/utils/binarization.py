import cv2
import numpy as np

# ---------------
# グローバル二値化
# ---------------
def binalize(src: np.ndarray, threshold: int = 127, Type=cv2.THRESH_BINARY):
    """
    単純閾値処理を行う関数

    Args:
        src (ndarray): 入力画像
        threshold (int): 閾値

    Returns:
        ret (int): 入力した閾値と同値
        dst (ndarray): 出力画像
    """
    new_img = src.copy()
    ret, dst = cv2.threshold(new_img, threshold, 255, Type)
    return ret, dst


# ---------------
# 大津の二値化
# ---------------
def otsu_binalize(src: np.ndarray):
    """
    大津の二値化処理を行う関数

    Arg:
        src (ndarray): 入力画像

    Returns:
        ret (int): 計算した閾値
        dst (ndarray): 出力画像
    """
    new_img = src.copy()
    ret, dst = cv2.threshold(new_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret, dst


# ---------------
# 適応的二値化
# ---------------
def adaptive_binalize(
    src: np.ndarray,
    method=cv2.ADAPTIVE_THRESH_MEAN_C,
    Type=cv2.THRESH_BINARY,
    block_size: int = 11,
    C: int = 2,
):
    """
    適応的閾値処理を行う関数

    Args:
        src (ndarray): 入力画像
    """
    new_img = src.copy()
    dst = cv2.adaptiveThreshold(new_img, 255, method, Type, block_size, C)
    return dst


def Twothresh_binalize(
    src: np.ndarray,
    LowerThreshold: int = 0,
    UpperThreshold: int = 128,
    Type=cv2.THRESH_BINARY,
):
    """2つの閾値を用いて画像を2値化処理する関数
    Args:
        src (np.ndarray): 3チャネルを持つ画像
        LowerThreshold (int, optional): 低い閾値. Defaults to 0.
        UpperThreshold (int, optional): 高い閾値. Defaults to 128.
        Type ([type], optional): 2値化処理のタイプ. Defaults to cv2.THRESH_BINARY.

    Returns:
        [type]: [description]
    """
    r, g, b = cv2.split(src)

    # for Red
    _, IMAGE_R_bw = binalize(r, LowerThreshold, Type)
    _, IMAGE_R__ = binalize(r, UpperThreshold, Type)
    IMAGE_R__ = cv2.bitwise_not(IMAGE_R__)
    # for Green
    _, IMAGE_G_bw = binalize(g, LowerThreshold, Type)
    _, IMAGE_G__ = binalize(g, UpperThreshold, Type)
    IMAGE_G__ = cv2.bitwise_not(IMAGE_G__)
    # for Blue
    _, IMAGE_B_bw = binalize(b, LowerThreshold, Type)
    _, IMAGE_B__ = binalize(b, UpperThreshold, Type)
    IMAGE_B__ = cv2.bitwise_not(IMAGE_B__)

    if PickupColor == 0:
        dst = IMAGE_R_bw * IMAGE_G__ * IMAGE_B__  # 画素毎の積を計算　⇒　赤色部分の抽出
    elif PickupColor == 1:
        dst = IMAGE_G_bw * IMAGE_B__ * IMAGE_R__  # 画素毎の積を計算　⇒　緑色部分の抽出
    elif PickupColor == 2:
        dst = IMAGE_B_bw * IMAGE_R__ * IMAGE_G__  # 画素毎の積を計算　⇒　青色部分の抽出
    elif PickupColor == 3:
        dst = IMAGE_R_bw * IMAGE_G_bw * IMAGE_B_bw  # 画素毎の積を計算　⇒　白色部分の抽出
    elif PickupColor == 4:
        dst = IMAGE_R__ * IMAGE_G__ * IMAGE_B__  # 画素毎の積を計算　⇒　黒色部分の抽出

    ret_low, ret_up = LowerThreshold, UpperThreshold

    return (ret_low, ret_up), dst


if __name__ == "__main__":
    # import sys
    # import os
    # print(os.getcwd())
    # sys.path.append('.')
    # sys.path.append('..')

    from matplotlib import pyplot as plt
    import numpy as np

    img_path = "./data/lena.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image is not read.")
    else:
        print("Image is read.")

    # print(img.shape)
    # -----------------
    # 二値化テスト
    # -----------------
    _, src = binalize(img)
    # _, src = otsu_binalize(img)
    # src = adaptive_binalize(img)

    # 画像をarrayに変換
    im_list = np.asarray(src)
    # 貼り付け
    plt.imshow(im_list, cmap="gray")
    # 表示
    plt.show()
