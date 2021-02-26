import sys, os

sys.path.append(".")
sys.path.append("..")

import cv2
import numpy as np
from ctypes import cdll

# from PIL import Image

import PySimpleGUI as sg
from src.config.config import cfg
from DobotDLL import DobotDllType as dType
from DobotFunction.Communication import Connect_Disconnect, Operation, _OneAction

# from DobotFunction.Camera import WebCam_OnOff


class Dobot_APP:
    def __init__(self):
        dll_path = cfg.DOBOT_DLL_DIR + os.sep + "DobotDll.dll"
        self.api = cdll.LoadLibrary(dll_path)
        self.CON_STR = {
            dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
            dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
            dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied",
        }
        self.pose_key = [
            "x",
            "y",
            "z",
            "r",
            "joint1Angle",
            "joint2Angle",
            "joint3Angle",
            "joint4Angle",
        ]
        self.current_pose = {}  # Dobotの現在の姿勢
        self.cam = None
        # --- エラーフラグ ---#
        self.connection = 1  # Connect: 0, DisConnect: 1, Err: -1
        self.act_err = 0  # State: 0, Err: -1
        # --- GUIの初期化 ---#
        self.layout = self.Layout()
        self.Window = self.main()

    """
    ----------------------
    Button Function
    ----------------------
    """

    def SetJointPose_click(self, pose: dict):
        """
        指定された作業座標系にアームの先端を移動させる関数
        関節座標系で移動

        Args:
            pose (dict):
                デカルト座標系もしくは関節座標系での移動先を示したリスト
                パラメータ数4個

        Returns:
            response (int):
                0 : 応答なし
                1 : 応答あり
        """
        return _OneAction(self.api, pose)

    def GetPose_UpdateWindow(self) -> dict:
        """
        姿勢を取得し、ウインドウを更新する関数

        Return:
            pose (dict):
                Dobotの現在の姿勢
        """

        pose = dType.GetPose(self.api)  # 現在のDobotの位置と関節角度を取得
        pose = dict(zip(self.pose_key, pose))  # 辞書形式に変換

        self.Window["-JointPose1-"].update(str(pose["joint1Angle"]))
        self.Window["-JointPose2-"].update(str(pose["joint2Angle"]))
        self.Window["-JointPose3-"].update(str(pose["joint3Angle"]))
        self.Window["-JointPose4-"].update(str(pose["joint4Angle"]))
        self.Window["-CoordinatePose_X-"].update(str(pose["x"]))
        self.Window["-CoordinatePose_Y-"].update(str(pose["y"]))
        self.Window["-CoordinatePose_Z-"].update(str(pose["z"]))
        self.Window["-CoordinatePose_R-"].update(str(pose["r"]))

        return pose

    """
    ----------------------
    GUI Layout
    ----------------------
    """

    def Layout(self):
        Connect = [sg.Button("Connect or Disconnect", key="-Connect-")]

        SetPose = [
            [
                sg.Button("Set pose", size=(7, 1), key="-SetJointPose-"),
                sg.Button("Set pose", size=(7, 1), key="-SetCoordinatePose-"),
            ],
            [
                sg.Text("J1", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPose1-"),
                sg.Text("X", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePose_X-"),
            ],
            [
                sg.Text("J2", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPose2-"),
                sg.Text("Y", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePose_Y-"),
            ],
            [
                sg.Text("J3", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPose3-"),
                sg.Text("Z", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePose_Z-"),
            ],
            [
                sg.Text("J4", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPose4-"),
                sg.Text("R", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePose_R-"),
            ],
        ]

        WebCamConnect = [
            [
                sg.Button("WEB CAM on/off", size=(15, 1), key="-SetWebCam-"),
                sg.Button("Preview Opened", size=(11, 1), key="-Preview-"),
                sg.Button("Snapshot", size=(7, 1), key="-Snapshot-"),
            ],
            [
                sg.InputCombo(
                    ("TOSHIBA_Web_Camera-HD", "Logicool_HD_Webcam_C270",),
                    size=(15, 1),
                    key="-WebCam_Name-",
                    readonly=True,
                ),
                sg.InputCombo(
                    (
                        "640x480",
                        "352x288",
                        "320x240",
                        "176x144",
                        "160x120",
                        "1280x720",
                        "1280x800",
                    ),
                    size=(11, 1),
                    key="-WebCam_FrameSize-",
                    readonly=True,
                ),
            ],
            [
                sg.Text("width", size=(4, 1)),
                sg.InputText(
                    "0",
                    size=(5, 1),
                    disabled=True,
                    justification="right",
                    key="-IMAGE_width-",
                ),
                sg.Text("height", size=(4, 1)),
                sg.InputText(
                    "0",
                    size=(5, 1),
                    disabled=True,
                    justification="right",
                    key="-IMAGE_height-",
                ),
                sg.Text("channel", size=(6, 1)),
                sg.InputText(
                    "0",
                    size=(5, 1),
                    disabled=True,
                    justification="right",
                    key="-IMAGE_channel-",
                ),
            ],
        ]

        layout = [
            Connect,
            [sg.Col(SetPose, size=(165, 136)),],
            [sg.Col(WebCamConnect),],
            [sg.Quit()],
        ]

        return layout

    """
    ----------------------
    GUI EVENT
    ----------------------
    """

    def Event(self, event, values):
        # Dobotの接続を行う
        if event == "-Connect-":
            # self.connection = self.Connect_Disconnect_click(self.connection, self.api)
            self.connection = Connect_Disconnect(
                self.connection, self.api, self.CON_STR
            )

            if self.connection == 0:
                # Dobotの現在の姿勢を画面上に表示
                self.current_pose = self.GetPose_UpdateWindow()

        elif event == "-SetJointPose-":
            # 移動後の関節角度を指定
            DestPose = [
                float(values["-JointPoseInput_1-"]),
                float(values["-JointPoseInput_2-"]),
                float(values["-JointPoseInput_3-"]),
                float(values["-JointPoseInput_4-"]),
            ]
            response = self.SetJointPose_click()
            print(response)

        # ------------------ #
        # WebCamに関するイベント #
        # ------------------ #
        elif event == "-SetWebCam-":
            device_num = WebCamOption(values["-WebCam_Name-"])
            if device_num is None:
                sg.popup("選択したデバイスは存在しません。", title="エラー")
                return

            self.cam = WebCam_OnOff(device_num, cam=self.cam)
            sg.popup("WebCameraに接続しました。", title="Camの接続")

    def main(self):
        return sg.Window(
            "Dobot",
            self.layout,
            default_element_size=(40, 1),
            background_color="grey90",
        )

    def loop(self):
        while True:
            event, values = self.Window.Read(timeout=10)
            if event == "Quit":
                break
            if event != "__TIMEOUT__":
                self.Event(event, values)


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

    if device_name == "TOSHIBA_Web_Camera-HD":
        device_num = 0
    elif device_name == "Logicool_HD_Webcam_C270":
        device_num = 1
    else:
        device_num = None

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


if __name__ == "__main__":
    # from DobotFunction.Camera import WebCam_OnOff
    # from ImageProcessing.binarization import GlobalThreshold

    # dll_path = cfg.DOBOT_DLL_DIR + os.sep + "DobotDll.dll"
    # print(dll_path)
    # api = cdll.LoadLibrary(dll_path)

    window = Dobot_APP()
    window.loop()

