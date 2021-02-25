import sys, os

sys.path.append(".")
sys.path.append("..")

import cv2


def WebCamOption(device_name):
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


def WebCam_OnOff(device_num, capture=None):
    """
    WebCameraを読み込む関数

    Parameter
    ---------
    device_num : int
        カメラデバイスを番号で指定
        0:PC内臓カメラ
        1:外部カメラ
    capture : OpenCV型
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
    if capture is None:  # カメラが接続されていないとき
        capture = cv2.VideoCapture(device_num)
        # カメラに接続できなかった場合
        if not capture.isOpened():
            return -1, None
        # 接続できた場合
        else:
            return 1, capture

    else:  # カメラに接続されていたとき
        capture.release()
        return 0, None


def Snapshot(capture):
    """
    WebCameraでスナップショットを撮影する関数
    capture : OpenCV型
        接続しているカメラ情報

    Return
    ------
    response : int
        0: 撮影できました。
        -1: 撮影できませんでした。

    frame : OpenCV型
        撮影した画像
    """
    if capture == None:
        return -1, None

    ret, frame = capture.read()  # 静止画像をGET
    if not capture.isOpened():
        return response, None

    response = 0
    return response, frame


if __name__ == "__main__":
    response, capture = WebCam_OnOff(device_num=0)
    if response == 1:
        Snapshot(capture=capture)
