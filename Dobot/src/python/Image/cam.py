import cv2
import argparse

def WebCam_OnOff(device_num: int, cam: cv2.VideoCapture = None):
    """
    WebCameraを読み込む関数

    Parameter
    ---------
    device_num : int
        カメラデバイスを番号で指定
        0:PC内臓カメラ
        1:外部カメラ
    cam : (cv2.VideoCapture)
        接続しているカメラ情報

    Return
    ------
    response: int
        動作終了を表すフラグ
        0: カメラを開放した
        1: カメラに接続した
        -1: エラー
    cam : OpenCV型
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
        cam.release()
        return 0, None



parser = argparse.ArgumentParser()
parser.add_argument("device_num", help='The number of the connected camera', type=int)
args = parser.parse_args()
device_num = args.device_num

Err, cam = WebCam_OnOff(device_num)