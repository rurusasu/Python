import sys, os

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import cv2
import numpy as np


def AutoGrayScale(img, clearly: bool = False, calc: str = "cv2") -> np.ndarray:
    """入力画像を自動的にグレースケール画像に変換する関数

    Args:
        img (np.ndarray):
            変換前の画像
        clearly (bool optional):
            ガウシアンフィルタを用いて入力画像のノイズを除去する．
            * True: 適用する
            * False: 適用しない: default
        calc (str, optional):
            グレースケール変換を行うための関数を指定.
            Defaults to "cv2".

    Returns:
        dst (np.ndarray):
            変換後の画像データ(Errorが発生した場合: None)

    """

    dst = img.copy()
    try:
        # ガウスフィルタをかけてノイズを除去する
        if clearly:
            dst = cv2.GaussianBlur(dst, (5, 5), 0)
        # 入力画像がRGBの時
        if len(dst.shape) > 2:
            dst = _GrayScale(dst, calc)
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
    # from DobotFunction.Camera import WebCam_OnOff, Snapshot, Preview
    # _, cam = WebCam_OnOff(0)
    # _, img = Snapshot(cam)
    # Preview(img, preview="plt")
    import json

    from PIL import Image
    from matplotlib import pyplot as plt
    from src.config.config import cfg

    # テスト画像の保存先
    save_path = cfg.GRAY_IMG_DIR
    clac_type = "cv2"

    # テストデータロード
    json_path = cfg.TEST_DIR + os.sep + "data.json"
    with open(json_path, mode="rt", encoding="utf_8") as f:
        datas = json.load(f)

    # テスト
    for data in datas["org_img"]:
        for key, value in data.items():
            if key == "path":
                # 画像を開いて numpy 配列に変換
                img = np.array(Image.open(value))
                # グレースケール化
                img = AutoGrayScale(img, calc=clac_type)

                img = Image.fromarray(img)
                img.save(save_path + os.sep + name + ".png")
            elif key == "name":
                name = value.rstrip(".png") + "_gray_" + clac_type

    # plt.imshow(img, cmap="gray")
    # plt.show()

