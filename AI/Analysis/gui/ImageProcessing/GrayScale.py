import sys, os

sys.path.append(".")
sys.path.append("..")

import cv2
import numpy as np


def GrayScale(img: np.ndarray, calc: str = "cv2") -> np.ndarray:
    """[summary]

    Args:
        img (np.ndarray):
            変換前の画像
        calc (str, optional):
            グレースケールを行うための関数を指定
            Defaults to "cv2".

    
        responce (int):
            1: 変換できた
            -1: 変換前の画像が入力されていない
        dst (np.ndarray):
            グレースケール化後の画像
    """
    if type(img) is np.ndarray:
        if calc == "cv2":
            dst = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return 1, dst
    else:
        return -1, None


if __name__ == "__main__":
    from DobotFunction.Camera import WebCam_OnOff, Snapshot, Preview

    _, cam = WebCam_OnOff(0)
    _, img = Snapshot(cam)

    # 閾値を用いて大域的二値化を行う
    _, img = GrayScale(img)
    Preview(img, preview="plt")
