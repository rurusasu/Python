import cv2
from typing import Type, Union

import numpy as np

from lib.utils.base_utils import Disassembly


class make_bin_img(object):
    img_elem = {
        "orig": np.array([]),
        "gray": np.array([]),
        "r": np.array([]),
        "g": np.array([]),
        "b": np.array([]),
    }
    bin_elem = {
        "gray": np.array([]),
        "r": np.array([]),
        "g": np.array([]),
        "b": np.array([]),
        "w": np.array([]),
        "bk": np.array([]),
    }
    bin_elem_rev = {
        "r_rev": np.array([]),
        "g_rev": np.array([]),
        "b_rev": np.array([]),
    }
    lower_thred = 0
    upper_thred = 0

    def __init__(self, src: np.ndarray, threshold: Type[Union[int, tuple]]) -> None:
        self.img_elem["orig"] = src
        if len(src.shape) == 1:
            self.img_elem["gray"] = src
        elif len(src.shape) == 3:
            self.img_elem["gray"] = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            [self.img_elem["r"], self.img_elem["g"], self.img_elem["b"]] = Disassembly(
                src
            )

        # 1つの閾値しか入力されていない場合
        if type(threshold) == int and (0 <= threshold <= 255):
            self.lower_thred = threshold
        # 複数の閾値が設定されている場合
        elif type(threshold) == tuple and (0 <= min(threshold), max(threshold) <= 255):
            self.lower_thred = min(threshold)
            self.upper_thred = max(threshold)
        else:
            raise ValueError("Input value is invalid.")

    def binarize(self, binary_type: str):
        """様々な二値化処理を行う関数

        Args:
            binary_type (str): 二値化処理タイプ．
            "Global": 大域的二値化処理
            "Otsu": 大津の二値化
            "Adaptive": 適応的二値化

        Returns:
            img_elem (dict): オリジナル画像，グレー画像，RGB要素ごとのグレー画像の要素が入った辞書．
            bin_elem (dict): img_elem のデータを二値化した画像の辞書．
            ret (int): 二値化に使用した閾値．エラーの場合は，'-1' を返す
        """
        # 2つの閾値を用いる場合とそうでない処理で条件分岐
        # なぜなら、2つの閾値を用いる場合は最大2個、そうでない処理の場合は最大3個の独立した閾値を持つため。
        ret = -1
        if binary_type == "Two_Thresh":
            self._two_thresh_binalize(
                LowerThreshold=self.lower_thred,
                UpperThreshold=self.upper_thred,
            )
        else:
            for i, color in enumerate(self.img_elem):
                if color == "orig":
                    continue

                if binary_type == "Global":
                    ret, dst = binalize(self.img_elem[color], self.lower_thred)
                elif binary_type == "Otsu":
                    ret, dst = otsu_binalize(self.img_elem[color])
                elif binary_type == "Adaptive":
                    dst = adaptive_binalize(self.img_elem[color])

                # bit 反転した画像を保存
                if color != "gray":
                    dst_rev = cv2.bitwise_not(dst)
                    self.bin_elem_rev[f"{color}_rev"] = dst_rev
                self.bin_elem[color] = dst

            self._w_and_bk_binarize()

        return self.img_elem, self.bin_elem, ret

    def _w_and_bk_binarize(self) -> None:
        # White color pickup
        w = self.bin_elem["r"] * self.bin_elem["g"] * self.bin_elem["b"]
        self.bin_elem["w"] = w

        # bk color pickup
        bk = (
            self.bin_elem_rev["r_rev"]
            * self.bin_elem_rev["g_rev"]
            * self.bin_elem_rev["b_rev"]
        )
        self.bin_elem["bk"] = bk

    def _two_thresh_binalize(
        self,
        LowerThreshold: int = 0,
        UpperThreshold: int = 128,
        Type=cv2.THRESH_BINARY,
    ) -> None:
        """2つの閾値を用いて画像を2値化処理する関数
        Args:
            LowerThreshold (int, optional): 低い閾値. Defaults to 0.
            UpperThreshold (int, optional): 高い閾値. Defaults to 128.
            Type ([type], optional): 2値化処理のタイプ. Defaults to cv2.THRESH_BINARY.
        """
        color_list = ["r", "g", "b"]

        for c in color_list:
            _, self.bin_elem[c] = binalize(self.img_elem[c], LowerThreshold, Type)
            _, tmp = binalize(self.img_elem["r"], UpperThreshold, Type)
            self.bin_elem_rev[f"{c}_rev"] = cv2.bitwise_not(tmp)

        # 画素毎の積を計算　⇒　赤色部分の抽出
        dst_r = (
            self.bin_elem["r"] * self.bin_elem_rev["g_rev"] * self.bin_elem_rev["b_rev"]
        )
        # 画素毎の積を計算　⇒　緑色部分の抽出
        dst_g = (
            self.bin_elem_rev["r_rev"] * self.bin_elem["g"] * self.bin_elem_rev["b_rev"]
        )
        # 画素毎の積を計算　⇒　青色部分の抽出
        dst_b = (
            self.bin_elem_rev["r_rev"] * self.bin_elem_rev["g_rev"] * self.bin_elem["b"]
        )
        # 画素毎の積を計算　⇒　白色部分の抽出
        self.bin_elem["w"] = (
            self.bin_elem["r"] * self.bin_elem["g"] * self.bin_elem["b"]
        )
        # 画素毎の積を計算　⇒　黒色部分の抽出
        self.bin_elem["bk"] = (
            self.bin_elem_rev["r_rev"]
            * self.bin_elem_rev["g_rev"]
            * self.bin_elem_rev["b_rev"]
        )

        # ほかの色も保存
        self.bin_elem["r"] = dst_r
        self.bin_elem["g"] = dst_g
        self.bin_elem["b"] = dst_b


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
