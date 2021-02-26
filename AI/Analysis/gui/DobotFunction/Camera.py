import sys, os

sys.path.append(".")
sys.path.append("..")

import cv2
import matplotlib.pyplot as plt
import numpy as np


def WebCamOption(device_name: str) -> int:
    """
    接続するWebCameraを選択する関数

    Parameter
    ---------
    device_name : int
        使用したいデバイス名を指定

    Return
    ------
    device_num : int
        名前が一致したデバイスに割り当てられた番号を返す
    """

    device_num = None
    if device_name == "TOSHIBA_Web_Camera-HD":
        device_num = 0
    if device_name == "Logicool_HD_Webcam_C270":
        device_num = 1

    return device_num


def WebCam_OnOff(device_num: int, cam: cv2.VideoCapture = None):
    """
    WebCameraを読み込む関数

    Parameter
    ---------
    device_num : int
        カメラデバイスを番号で指定
        0:PC内臓カメラ
        1:外部カメラ
    cam : OpenCV型
        接続しているカメラ情報

    Return
    ------
    response: int
        動作終了を表すフラグ
        0: カメラを開放した
        1: カメラに接続した
        -1: エラー
    capture : OpenCV型
        接続したデバイス情報を返す
    """
    if cam is None:  # カメラが接続されていないとき
        cam = cv2.VideoCapture(device_num)
        # カメラに接続できなかった場合
        if not cam.isOpened():
            return -1, None
        # 接続できた場合
        else:
            return 1, cam

    else:  # カメラに接続されていたとき
        capture.release()
        return 0, None


def Snapshot(cam: cv2.VideoCapture = None) -> np.ndarray:
    """
    WebCameraでスナップショットを撮影する関数
    cam : cv2.VideoCapture
        接続しているカメラ情報
        default : None

    Return
    ------
    response : int
        1: 撮影できました。
        -1: 撮影できませんでした。

    img : np.ndarray
        撮影した画像
    """
    # カメラが接続されていない場合
    if cam == None:
        return -1, None

    ret, img = cam.read()  # 静止画像をGET
    # 静止画が撮影できた場合
    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        img = np.array(img)
        return 1, img
    # 撮影できなかった場合
    else:
        return -1, None


def Preview(img: np.ndarray = None, window_name: str = "frame", preview: str = "cv2"):
    """
    webカメラの画像を表示する関数

    Parameters
    ----------
    img : ndarray型
        画像のピクセル値配列
        default : None
    window_name : str
        画像を表示する時のウインドウ名
        default : "frame"
    preview : str
        画像をウインドウ上に表示するときに使用するパッケージ名
        OpenCV の imshow を使用する場合 : "cv2" (default)
        Matplotlib の plt.show を使用する場合: "plt"

    Returns
    -------
    response : int
        画像表示の可否を返す
        1: 表示できた。
        -1: 表示できない。
    """
    # 画像が入力されている場合
    if type(img) is np.ndarray:
        # 画像を OpenCV でウインドウ上に表示する
        if preview == "cv2":
            cv2.imshow(window_name, img)
            return 1
        # 画像を Matplotlib で ウインドウ上に表示する
        elif preview == "plt":
            # グレースケール画像の場合
            if len(img.shape) == 2:
                plt.imshow(img, cmap="gray")
            # RGB画像の場合
            elif len(img.shape) == 3:
                plt.imshow(img)
            plt.show()
            return 1
        # 表示に使うパッケージの選択が不適切な場合
        else:
            return -1
    # 画像が入力されていない場合
    else:
        return -1


if __name__ == "__main__":
    response, cam = WebCam_OnOff(device_num=0)
    if response == 1:
        response, img = Snapshot(cam=cam)
        if response == 1:
            Preview(img, preview="plt")

