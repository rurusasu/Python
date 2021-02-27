import sys, os

sys.path.append(".")
sys.path.append("..")

import cv2
import numpy as np


def AutoGrayScale(img, calc: str = "cv2") -> np.ndarray:
    """入力画像を自動的にグレースケール画像に変換する関数

    Args:
        img (np.ndarray):
            変換前の画像
        calc (str, optional):
            グレースケール変換を行うための関数を指定.
            Defaults to "cv2".

    Returns:
        np.ndarray: 変換後の画像データ(Errorが発生した場合: None)

    """
    # if type(img) is np.ndarray:
    try:
        # 入力画像がRGBの時
        if len(img.shape) > 2:
            dst = _GrayScale(img, calc)
    except Exception as e:
        print("GrayScaleError:", e)
        return None
    else:
        return dst


def _GrayScale(img: np.ndarray, calc: str = "cv2") -> np.ndarray:
    """入力画像をグレースケール画像に変換する関数

    Args:
        img (np.ndarray):
            変換前の画像
        calc (str, optional):
            グレースケールを行うための関数を指定
            "cv2": cv2.cvtColor で変換

    Return:
        dst (np.ndarray):
            グレースケール化後の画像
    """
    if calc == "cv2":
        dst = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return dst
    else:
        return None


if __name__ == "__main__":
    from DobotFunction.Camera import WebCam_OnOff, Snapshot, Preview

    _, cam = WebCam_OnOff(0)
    _, img = Snapshot(cam)

    # 閾値を用いて大域的二値化を行う
    img = AutoGrayScale(img)
    # img = _GrayScale(img)
    Preview(img, preview="plt")
