# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

import PySimpleGUI as sg
import numpy as np

import DobotDllType as dType
from common.DobotFunction import initDobot, Operation, OneAction
from ctypes import cdll

import time
from timeout_decorator import timeout, TimeoutError

import cv2
from PIL import Image

def __filePath__(file_name):
    path = './data/' + str(file_name)
    return path

# ----- The callback function ----- #
class Dobot_APP:
    def __init__(self):
        self.api = cdll.LoadLibrary('DobotDll.dll')
        self.CON_STR = {
            dType.DobotConnect.DobotConnect_NoError:  'DobotConnect_NoError',
            dType.DobotConnect.DobotConnect_NotFound: 'DobotConnect_NotFound',
            dType.DobotConnect.DobotConnect_Occupied: 'DobotConnect_Occupied'
        }
        self.CurrentPose = None
        self.queue_index = 0
        self.capture = None
        self.IMAGE_original = None
        #--- エラーフラグ ---#
        self.connection = 0 # Connect:1, DisConnect:0
        self.cammera_connection = 0 # Connect:1, DisConnect:0
        self.prevWindowOpened = 0 # Opened:1, NotOpened:0
        self.DOBOT_err = 0  # Error occurred:1, No error:0
        self.Input_err = 0  # Error occurred:1, No error:0
        #--- GUIの初期化 ---#
        self.layout = self.Layout()
        self.Window = self.main()

    def Connect_click(self):
        """
        Dobotを接続する関数

        Returns
        -------
        result : int
            0 : 接続できなかった場合
            1 : 接続できた場合
            2 : すでに接続されていた場合

        """
        # Dobotがすでに接続されていた場合
        if self.connection == 1:
            return 2
        # Dobot Connect
        state = dType.ConnectDobot(self.api, "", 115200)[0] # ConectDobot(const char* pointName, int baudrate)
        if (state != dType.DobotConnect.DobotConnect_NoError):
            return 0 # Dobotに接続できなかった場合
        else:
            dType.SetCmdTimeout(self.api, 3000)
            return 1 # Dobotに接続できた場合 

    def Disconnect_click(self):
        """
        Dobotとの接続を切断する関数

        Returns
        -------
        result : int
            result : int
            0 : 接続されていない場合
            1 : 切断できた場合
        """
        # Dobotが接続されていない場合
        if self.connection == 0:
            return 0
        # DobotDisconnect
        dType.DisconnectDobot(self.api)
        return 1

    def GetPose_click(self):
        """
        デカルト座標系と関節座標系でのDobotの姿勢を返す関数

        Returns
        -------
        response : int
            0 : 応答なし
            1 : 応答あり

        PoseParams : list
            Dobotの姿勢を格納したリスト
        """
        response = 0 # Dobotからの応答 1:あり, 0:なし
        if self.connection == 0:
            self.DOBOT_err = 1
            return response

        timeout(5)
        try:
            pose = dType.GetPose(self.api)
        except TimeoutError:
            self.DOBOT_err = 1
            return response
        response = 1
        self.CurrentPose = pose # 現在の姿勢をグローバル変数に保存
        return response, pose

    def SetJointPose_click(self, pose):
        """
        指定された作業座標系にアームの先端を移動させる関数
        関節座標系で移動

        Parameters
        ----------
        pose : list
            デカルト座標系もしくは関節座標系での移動先を示したリスト
            パラメータ数4個

        Returns
        -------
        response : int
            0 : 応答なし
            1 : 応答あり
        """
        response = 0 # Dobotからの応答 1:あり, 0:なし
        if self.connection == 0 or self.DOBOT_err == 1:
            self.DOBOT_err = 1
            return response

        timeout(5)
        try:
            dType.SetPTPCmd(self.api,
                            dType.PTPMode.PTPMOVJANGLEMode,
                            pose[0],
                            pose[1],
                            pose[2],
                            pose[3],
                            self.queue_index)
        except TimeoutError:
            self.DOBOT_err = 1
            return response
        response = 1
        return response

    def SetCoordinatePose_click(self, pose):
        """
        指定された作業座標系にアームの先端を移動させる関数
        デカルト座標系で移動

        Parameters
        ----------
        pose : list
            デカルト座標系もしくは関節座標系での移動先を示したリスト
            パラメータ数4個

        Returns
        -------
        response : int
            0 : 応答なし
            1 : 応答あり
            2 : 姿勢が保存されていない
        """
        response = 0  # Dobotからの応答 1:あり, 0:なし
        if self.connection == 0 or self.DOBOT_err == 1:
            self.DOBOT_err = 1
            return response

        timeout(5)
        try:
            dType.SetPTPCmd(self.api,
                            dType.PTPMode.PTPMOVJXYZMode,
                            pose[0],
                            pose[1],
                            pose[2],
                            pose[3],
                            self.queue_index)
        except TimeoutError:
            self.DOBOT_err = 1
            return response
        response = 1
        return response

    def WebCam_OnOff_click(self, device_num):
        """
        WebCameraを読み込む関数

        Parameter
        ---------
        device_num : int
            カメラデバイスを番号で指定
            0:PC内臓カメラ
            1:外部カメラ

        Return
        ------
        camera_connection : int
            カメラに接続できたかを確認する関数
            0 : 接続してない
            1 : 接続した
        """
        camera_connection = 0 # カメラの応答 1:あり, 0:なし
        if self.cammera_connection == 0: # カメラが接続されていない場合
            self.capture = cv2.VideoCapture(device_num)
            if not self.capture.isOpened(): # カメラに接続できなかった場合
                return camera_connection
            else: # カメラに接続できた場合
                camera_connection = 1
                return camera_connection

        else: # カメラが接続されていた場合
            self.capture.release() # カメラを解放する
            self.capture = None
            camera_connection = 2
            return camera_connection

    def PreviewOpened_click(self, window_name='frame', delay=1):
        """
        webカメラの画像を表示する関数

        Parameters
        ----------
        window_name

        Returns
        -------
        response : int
            画像表示の可否を返す
            0: 表示できない。
            1: 表示できた。
        """
        response = 0
        if (self.cammera_connection == 0) or (self.capture == None):
            return response

        if self.prevWindowOpened == 0: # prevWindowが閉じているとき
            while True:
                ret, frame = self.capture.read()
                if ret:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(delay) & 0xFF == ord('e'):
                        break
                else:
                    print('未知のエラーです。')
            cv2.destroyWindow(window_name)
            response = 1
            return response

    def ObjectDetectionInSnapshot(self):
        """
        WebCameraでスナップショットを撮影する関数

        Return
        ------
        response : int
            0: 撮影できませんでした。
            1: 撮影できました。
        """
        response = 0
        if (self.cammera_connection == 0) or (self.capture == None):
            self.IMAGE_original = None
            return response

        ret, frame = self.capture.read() # 静止画像をGET
        
        if not self.capture.isOpened():
            self.IMAGE_original = None
            return response

        cv2.imshow('Snapshot', frame) # スナップショットを表示する
        self.IMAGE_original = frame
        response = 1
        return response

    def Gray2RGB(self):
        response = 0
        if self.IMAGE_original is None:  # 画像の元データが存在しない場合
            return response
        
        


    def GlobalThreshold(self, threshold=127, Type=cv2.THRESH_BINARY):
        """
        単純な閾値処理

        Parameters
        ----------
        threshold : flaot
            2値化するときの閾値
        type
            閾値の処理方法
            ・cv2.THRESH_BINARY
            ・cv2.THRESH_BINARY_INV
            ・cv2.THRESH_TRUNC
            ・cv2.THRESH_TOZERO
            ・cv2.THRESH_TOZERO_INV
        """
        response = 0
        if self.IMAGE_original is None:  # 画像の元データが存在しない場合
            return response
        
        result, img = self.AnyImage2RGB()
        if result == 0 or img == None:
            return response
            
        ret, thresh = cv2.threshold(img, threshold, 255, Type)

        return ret



    def AnyImage2RGB(self):
        """
        画像の変換を行う関数

        Return
        ------
        response : int
            0: 画像の元データが存在しません。
            1: 変換できました。
        new_image
            変換した画像
        """
        response = 0
        if self.IMAGE_original is None:  # 画像の元データが存在しない場合
            return response, None

        new_image = self.IMAGE_original.copy()
        #-------------------------------#
        #          Gray Scale           #
        #-------------------------------#
        if new_image.ndim == 2: # モノクロ
            gamma22LUT = np.array([pow(x/255.0, 2.2)
                                   for x in range(256)], dtype='float32')
            img_barL = cv2.LUT(new_image, gamma22LUT)
            img_grayL = cv2.cvtColor(img_barL, cv2.COLOR_BGR2GRAY)
            img_gray = pow(img_grayL, 1.0/2.2) * 255
            new_image = img_gray
        #-------------------------------#
        #             RGB               #
        #-------------------------------#
        elif new_image.shape[2] == 3: # RGB画像
            b, g, r = cv2.split(new_image)
            new_image = cv2.merge((r, g, b))
            new_image = new_image[:,:,::-1]
        #-------------------------------#
        #             RGBA              #
        #-------------------------------#
        elif new_image.shape[2] == 4:
            b, g, r, a = cv2.split(new_image)
            new_image = cv2.merge((r,g,b,a))
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2RGB)
        
        response = 1
        cv2.imshow('Convert', new_image)

        return response, new_image


    def SaveOriginal_click(self, CON_STR, file_name):
        x_roop = 10
        y_roop = 20
        z_roop = 1

        file_name = __filePath__(file_name)
        print(file_name + 'にデータを保存します。')

        initPOS = dType.GetPose(api)
        #-----------------------------
        # 以下Z軸方向の動作
        #-----------------------------
        for i in range(0, z_roop):
            print('第' + str(i + 1) + 'ステップ目')
            #Operation(api, file_name, 'z', -i*10, initPOS)

            #-------------------------
            # 以下Y軸方向の動作
            #-------------------------
            for j in range(0, y_roop):
                Operation(api, file_name, 'y', 10)

                #-------------------------
                # 以下X軸方向の動作
                #-------------------------
                if j % 2 == 0:
                    for k in range(0, x_roop + 1):
                        #Async Motion Params Setting
                        Operation(api, file_name, 'x', 10)
                else:
                    for k in range(0, x_roop + 1):
                        #Async Motion Params Setting
                        Operation(api, file_name, 'x', -10)

        print('データ取得が終了しました。')

    def SaveValidation_click(CON_STR, file_name):
        x_roop = 100
        y_roop = 100
        z_roop = 2

        file_name = __filePath__(file_name)
        print(file_name + 'にデータを保存します。')

        initPOS = dType.GetPose(api)
        #-----------------------------
        # 以下Z軸方向の動作
        #-----------------------------
        for i in range(0, z_roop):
            print('第' + str(i + 1) + 'ステップ目')
            Operation(api, file_name, 'z', -0.5*i, initPOS)

            #-------------------------
            # 以下Y軸方向の動作
            #-------------------------
            for j in range(0, y_roop):
                Operation(api, file_name, 'x', 0.5)

                #-------------------------
                # 以下X軸方向の動作
                #-------------------------
                if j % 2 == 0:
                    for k in range(0, x_roop + 1):
                        #Async Motion Params Setting
                        Operation(api, file_name, 'y', 0.5)
                else:
                    for k in range(0, x_roop + 1):
                        #Async Motion Params Setting
                        Operation(api, file_name, 'y', -0.5)

        print('testデータ取得が終了しました。')

    def DobotAct(self, x_pos, y_pos, z_pos):
        _OneAction(api, x=x_pos, y=y_pos, z=z_pos)


    """
    ----------------------
    GUI Layout
    ----------------------
    """
    def Layout(self):
        # ----- Menu Definition ----- #
        menu_def = [['File', ['Open', 'Save', 'Exit', 'Properties']],
                    ['Edit', []],
                    ['Help'],]

        # ----- Column Definition ----- #
        """
        JointPose = [
            [sg.Text('J1', size=(3, 1)), sg.Text('', size=(3, 1), key='-JointPose1-')],
            [sg.Text('J2', size=(3, 1)), sg.Text('', size=(3, 1), key='-JointPose2-')],
            [sg.Text('J3', size=(3, 1)), sg.Text('', size=(3, 1), key='-JointPose3-')],
            [sg.Text('J4', size=(3, 1)), sg.Text('', size=(3, 1), key='-JointPose4-')],
        ]
        """
        CurrentPose = [
            [sg.Button('Get Pose', size=(7, 1), key='-GetPose-')],
            [sg.Text('J1', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-JointPose1-'),
             sg.Text('X', size=(2, 1)),  sg.InputText('', size=(5, 1), disabled=True, key='-CoordinatePose_X-')],
            [sg.Text('J2', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-JointPose2-'),
             sg.Text('Y', size=(2, 1)),  sg.InputText('', size=(5, 1), disabled=True, key='-CoordinatePose_Y-')],
            [sg.Text('J3', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-JointPose3-'),
             sg.Text('Z', size=(2, 1)),  sg.InputText('', size=(5, 1), disabled=True, key='-CoordinatePose_Z-')],
            [sg.Text('J4', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-JointPose4-'),
             sg.Text('R', size=(2, 1)),  sg.InputText('', size=(5, 1), disabled=True, key='-CoordinatePose_R-')],
        ]
        
        SetJointPose = [
            [sg.Button('Set pose', key='-SetJointPose-')],
            [sg.Text('J1', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_1-')],
            [sg.Text('J2', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_2-')],
            [sg.Text('J3', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_3-')],
            [sg.Text('J4', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_4-')],
        ]

        SetCoordinatePose = [
            [sg.Button('Set pose', key='-SetCoordinatePose-')],
            [sg.Text('X', size=(2, 1)), sg.InputText('', size=(5, 1), key='-CoordinatePoseInput_X-')],
            [sg.Text('Y', size=(2, 1)), sg.InputText('', size=(5, 1), key='-CoordinatePoseInput_Y-')],
            [sg.Text('Z', size=(2, 1)), sg.InputText('', size=(5, 1), key='-CoordinatePoseInput_Z-')],
            [sg.Text('R', size=(2, 1)), sg.InputText('', size=(5, 1), key='-CoordinatePoseInput_R-')],
        ]
        
        WebCam = [
            [sg.Button('WEB CAM on/off', key='-SetWebCam-'),
             sg.Button('Preview opened/closed', key='-Preview-'),],
            [sg.Button('Snapshot', key='-Snapshot-')],
            [sg.Text('width', size=(4, 1)), sg.InputText('0', size=(5, 1), disabled=True, key='-IMAGE_width-'),
             sg.Text('height', size=(5, 1)), sg.InputText('0', size=(5, 1), disabled=True, key='-IMAGE_height-'), 
             sg.Text('channel', size=(5, 1)), sg.InputText('0', size=(5, 1), disabled=True, key='-IMAGE_channel-'),]
        ]

        ColorOfObject = [
            [sg.Radio('R', group_id='color', background_color='grey59', text_color='red', key='-color_1-'),
             sg.Radio('G', group_id='color', background_color='grey59', text_color='green', key='-color_2-'),
             sg.Radio('B', group_id='color', background_color='grey59', text_color='blue', key='-color_3-'),
             sg.Radio('W', group_id='color', background_color='grey59', text_color='snow', key='-color_4-'),
             sg.Radio('Bk', group_id='color', default=True, background_color='grey59', text_color='grey1', key='-color_5-')],
        ]

        Threshold = [
            [sg.Checkbox('Global Threshold', key='-GlobalThreshold-'),
             sg.Checkbox('Adaptive Threshold', key='-AdaptiveThreshold-'),]
        ]

        saveOrg = [
            [sg.Text('OriginalData')],
            [sg.Text('FileName'), 
             sg.InputText('data.csv', size=(13, 1), key='-orgSave-')],
            [sg.Button('SaveOriginal', key='-SaveOriginal-')],
        ]

        saveVal = [
            [sg.Text('ValidationData')],
            [sg.Text('FileName'),
             sg.InputText('val.csv', size=(13, 1), key='-valSave-')],
            [sg.Button('SaveValidation', key='-SaveValidation-')],
        ]

        inputPoint = [
            [sg.Text('X座標'), sg.Input(size=(5, 1), key='-x-')],
            [sg.Text('Y座標'), sg.Input(size=(5, 1), key='-y-')],
            [sg.Text('Z座標'), sg.Input(size=(5, 1), key='-z-')],
            [sg.Button('ACT', key='-ACT-')]
        ]

        layout = [
            [sg.Text('Dobotを接続する')], 
            [sg.Button('Conect to DOBOT', key='-Connect-')],
            [sg.Button('Disconnect to DOBOT', key='-Disconnect-')],
            [sg.Col(CurrentPose), sg.Col(SetJointPose), sg.Col(SetCoordinatePose)],
            [sg.Col(WebCam)],
            [sg.Frame('Color of object', ColorOfObject, background_color='grey59')],
            [sg.Col(Threshold)],
            [sg.Frame('Save', 
                [[sg.Column(saveOrg)],
                 [sg.Column(saveVal)],])],
            [sg.Frame('移動座標', inputPoint)],
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
        if event == '-Connect-':
            result = self.Connect_click()
            if result == 0:
                sg.popup('Dobotに接続できません。', title='Dobotの接続')
                self.connection = 0
                return
            elif result == 1:
                sg.popup('Dobotに接続しました。', title='Dobotの接続')
                self.connection = 1
                return
            else:
                sg.popup('Dobotに接続しています。', title='Dobotの接続')
                return

        # Dobotの切断を行う
        elif event == '-Disconnect-':
            result = self.Disconnect_click()
            if result == 0:
                sg.popup('Dobotに接続できません。', title='Dobotの接続')
                return
            elif result == 1:
                sg.popup('Dobotの接続を切断しました。', title='Dobotの接続')
                self.connection = 0
                return

        # Dobotの現在の姿勢を取得し表示する
        elif event == '-GetPose-':
            if self.connection == 0:
                sg.popup('Dobotは接続していません。', title='Dobotの接続')
                return
            else:
                response, pose = self.GetPose_click()
                if response == 0:
                    sg.popup('Dobotからの応答がありません。', title='Dobotの接続')
                    return
                else:
                    print(values)
                    self.Window['-JointPose1-'].update(str(pose[4]))
                    self.Window['-JointPose2-'].update(str(pose[5]))
                    self.Window['-JointPose3-'].update(str(pose[6]))
                    self.Window['-JointPose4-'].update(str(pose[7]))
                    self.Window['-CoordinatePose_X-'].update(str(pose[0]))
                    self.Window['-CoordinatePose_Y-'].update(str(pose[1]))
                    self.Window['-CoordinatePose_Z-'].update(str(pose[2]))
                    self.Window['-CoordinatePose_R-'].update(str(pose[3]))
                    return

        elif event == '-SetJointPose-':
            response = 0  # Dobotからの応答 1:あり, 0:なし
            if self.connection == 0:
                sg.popup('Dobotに接続していません。', title='Dobotの接続')
                return
            else:
                if ((values['-JointPoseInput_1-'] is '') and (values['-JointPoseInput_2-'] is '') and (values['-JointPoseInput_3-'] is '') and (values['-JointPoseInput_4-'] is '')):
                    sg.popup('移動先が入力されていません。', title='入力不良')
                    self.Input_err = 1
                    return

                # 入力姿勢の中に''があるか判定
                if ((values['-JointPoseInput_1-'] is '') or (values['-JointPoseInput_2-'] is '') or (values['-JointPoseInput_3-'] is '') or (values['-JointPoseInput_4-'] is '')):
                    response, CurrentPose = self.GetPose_click()
                    if response == 0:  # GetPoseできなかった時
                        self.DOBOT_err = 1
                        return
                    else:
                        if values['-JointPoseInput_1-'] is '':
                            values['-JointPoseInput_1-'] = CurrentPose[4]
                        if values['-JointPoseInput_2-'] is '':
                            values['-JointPoseInput_2-'] = CurrentPose[5]
                        if values['-JointPoseInput_3-'] is '':
                            values['-JointPoseInput_3-'] = CurrentPose[6]
                        if values['-JointPoseInput_4-'] is '':
                            values['-JointPoseInput_4-'] = CurrentPose[7]

                # 移動後の関節角度を指定
                DestPose = [
                    float(values['-JointPoseInput_1-']),
                    float(values['-JointPoseInput_2-']),
                    float(values['-JointPoseInput_3-']),
                    float(values['-JointPoseInput_4-']),
                ]

                response_2 = self.SetJointPose_click(DestPose)
                if response_2 == 0:
                    sg.popup('Dobotからの応答がありません。', title='Dobotの接続')
                    return
                else: return

        elif event == '-SetCoordinatePose-':
            response = 0  # Dobotからの応答 1:あり, 0:なし
            if self.connection == 0:
                sg.popup('Dobotに接続していません。', title='Dobotの接続')
                return
            else:
                if ((values['-CoordinatePoseInput_X-'] is '') and (values['-CoordinatePoseInput_Y-'] is '') and (values['-CoordinatePoseInput_Z-'] is '') and (values['-CoordinatePoseInput_R-'] is '')):
                    sg.popup('移動先が入力されていません。', title='入力不良')
                    self.Input_err = 1
                    return

                # 入力姿勢の中に''があるか判定
                if ((values['-CoordinatePoseInput_X-'] is '') or (values['-CoordinatePoseInput_Y-'] is '') or (values['-CoordinatePoseInput_Z-'] is '') or (values['-CoordinatePoseInput_R-'] is '')):
                    response, CurrentPose = self.GetPose_click()
                    if response == 0:  # GetPoseできなかった時
                        self.DOBOT_err = 1
                        return
                    else:
                        if values['-CoordinatePoseInput_X-'] is '':
                            values['-CoordinatePoseInput_X-'] = CurrentPose[0]
                        if values['-CoordinatePoseInput_Y-'] is '':
                            values['-CoordinatePoseInput_Y-'] = CurrentPose[1]
                        if values['-CoordinatePoseInput_Z-'] is '':
                            values['-CoordinatePoseInput_Z-'] = CurrentPose[2]
                        if values['-CoordinatePoseInput_R-'] is '':
                            values['-CoordinatePoseInput_R-'] = CurrentPose[3]

                # 移動後の関節角度を指定
                DestPose = [
                    float(values['-CoordinatePoseInput_X-']),
                    float(values['-CoordinatePoseInput_Y-']),
                    float(values['-CoordinatePoseInput_Z-']),
                    float(values['-CoordinatePoseInput_R-']),
                ]

                response_2 = self.SetCoordinatePose_click(DestPose)
                if response_2 == 0:
                    sg.popup('Dobotからの応答がありません。', title='Dobotの接続')
                    return
                else:
                    return

        elif event == '-SetWebCam-':
            result = self.WebCam_OnOff_click(0)
            if result == 0:
                sg.popup('WebCameraに接続できません。', title='Camの接続')
                self.cammera_connection = 0
                return
            elif result == 1:
                sg.popup('WebCameraに接続しました。', title='Camの接続')
                self.cammera_connection = 1
                return
            else:
                sg.popup('WebCameraを解放しました。', title='Camの接続')
                self.cammera_connection = 0
                return

        elif event == '-Preview-':
            if self.cammera_connection == 0:
                sg.popup('WebCameraが接続されていません。', title='Camの接続')
                return
            
            result = self.PreviewOpened_click()
            if result == 0:
                sg.popup('WebCameraが接続されていません。', title='画像の表示')
                return
            else:
                sg.popup('WebCameraの画像を閉じました。', title='画像の表示')
                return

        elif event == '-Snapshot-':
            if self.cammera_connection == 0:
                sg.popup('WebCameraが接続されていません。', title='Camの接続')
                return
            result = self.ObjectDetectionInSnapshot()
            if result == 0:
                sg.popup('スナップショットを撮影できませんでした。', title='スナップショット')
                return
            else:
                sg.popup('スナップショットを撮影しました。', title='スナップショット')
                #print(self.IMAGE_original.shape)
                [y, x, z] = self.IMAGE_original.shape
                print(self.IMAGE_original.ndim)
                self.Window['-IMAGE_width-'].update(str(x))
                self.Window['-IMAGE_height-'].update(str(y))
                self.Window['-IMAGE_channel-'].update(str(z))

                image = self.IMAGE_Conv()
                sg.popup('画像の変換ができませんでした。', title='変換')
                return

        elif event is '-SaveOriginal-':
            if CON_STR == None:
                print('Dobotに接続していません。')
            else:
                self.SaveOriginal_click(CON_STR, values['-orgSave-'])
        elif event == '-SaveValidation-':
            if CON_STR == None:
                print('Dobotに接続していません。')
            else:
                self.SaveValidation_click(CON_STR, values['-valSave-'])
        elif event == '-ACT-':
            if CON_STR == None:
                print('Dobotに接続していません。')
            else:
                DobotAct(values['-x-'], values['-y-'], values['-z-'])
    
    def main(self):
        return sg.Window('Dobot', self.layout, default_element_size=(40, 1))

    def loop(self):
        while True:
            event, values = self.Window.Read(timeout=10)
            if event is 'Quit':
                break
            if event != '__TIMEOUT__':
                self.Event(event, values)
            




# ボタンを押したときのイベントとボタンが返す値を代入
#event, values = window.Read()

#CON_STR = Dobot()

if __name__ == '__main__':
    """
    window = Dobot_APP()
    Read = window.main()

    while True:
        #event, values = Read.Read(timeout=10)
        event, values = Read.Read(timeout=10)
        if event is 'Quit':
            break
        elif event != '__TIMEOUT__':
            window.Event(event, values)
    """
    window = Dobot_APP()
    window.loop()

