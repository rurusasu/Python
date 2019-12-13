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
        self.IMAGE_original = None  # 撮影した生画像
        self.IMAGE_original2 = None # 変換後の画像
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

    def Snapshot(self):
        """
        WebCameraでスナップショットを撮影する関数

        Return
        ------
        response : int
            0: 撮影できませんでした。
            1: 撮影できました。
        frame : OpenCV型
            撮影した画像
        """
        response = 0
        if (self.cammera_connection == 0) or (self.capture == None):
            self.IMAGE_original = None
            return response, None

        ret, frame = self.capture.read() # 静止画像をGET
        
        if not self.capture.isOpened():
            self.IMAGE_original = None
            return response, None

        response = 1
        return response, frame

    def AnyImage2RGB(self, img):
        """
        画像の変換を行う関数

        Parameters
        ----------
        img : OpenCV型
            変換前の画像

        Return
        ------
        response : int
            0: 画像の元データが存在しません。
            1: 変換できました。
        new_image
            変換した画像
        """
        response = 0
        if img is None:  # 画像の元データが存在しない場合
            return response, None

        new_image = img.copy()
        #-------------------------------#
        #          Gray Scale           #
        #-------------------------------#
        if new_image.ndim == 2:  # モノクロ
            gamma22LUT = np.array([pow(x/255.0, 2.2)
                                   for x in range(256)], dtype='float32')
            img_barL = cv2.LUT(new_image, gamma22LUT)
            img_grayL = cv2.cvtColor(img_barL, cv2.COLOR_BGR2GRAY)
            img_gray = pow(img_grayL, 1.0/2.2) * 255
            new_image = img_gray
        #-------------------------------#
        #             RGB               #
        #-------------------------------#
        elif new_image.shape[2] == 3:  # RGB画像
            b, g, r = cv2.split(new_image)
            new_image = cv2.merge((r, g, b))
            new_image = new_image[:, :, ::-1]
        #-------------------------------#
        #             RGBA              #
        #-------------------------------#
        elif new_image.shape[2] == 4:
            b, g, r, a = cv2.split(new_image)
            new_image = cv2.merge((r, g, b, a))
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2RGB)

        response = 1
        #cv2.imshow('Convert', new_image)
        return response, new_image

    def GlobalThreshold(self, img, gaussian=False, threshold=127, Type=cv2.THRESH_BINARY):
        """
        画素値が閾値より大きければある値(白色'255')を割り当て，そうでなければ別の値(黒色)を割り当てる。
        img is NoneならNoneを、変換に成功すれば閾値処理された2値画像を返す。

        Parameters
        ----------
        img : OpenCV型
            変換前の画像データ
        gaussian : True or False
            ガウシアンフィルタを適応するか選択できる。
        threshold : flaot
            2値化するときの閾値
        Type
            閾値の処理方法
            ・cv2.THRESH_BINARY
            ・cv2.THRESH_BINARY_INV
            ・cv2.THRESH_TRUNC
            ・cv2.THRESH_TOZERO
            ・cv2.THRESH_TOZERO_INV
        """
        if img is None:  # 画像の元データが存在しない場合
            return None
        if gaussian:
            img = cv2.GaussianBlur(img, (5, 5), 0)


        ret, thresh = cv2.threshold(img, threshold, 255, Type)
        return thresh
    
    def OtsuThreshold(self, img, gaussian=False):
        """
        入力画像が bimodal image (ヒストグラムが双峰性を持つような画像)であることを仮定すると、
        そのような画像に対して、二つのピークの間の値を閾値として選べば良いと考えることであろう。これが大津の二値化の手法である。
        双峰性を持たないヒストグラムを持つ画像に対しては良い結果が得られないことになる。

        Parameters
        ----------
        img : OpenCV型
            変換前の画像データ
        gaussian : True or False
            ガウシアンフィルタを適応するか選択できる。
        """
        if img is None:  # 画像の元データが存在しない場合
            return None
        # 画像のチャンネル数が2より大きい場合
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ガウシアンフィルタで前処理を行う場合
        if gaussian:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        return img

    def AdaptiveThreshold(self, img, gaussian=False, method=cv2.ADAPTIVE_THRESH_MEAN_C, Type=cv2.THRESH_BINARY, block_size=11, C=2):
        """
        適応的閾値処理では，画像の小領域ごとに閾値の値を計算する．
        そのため領域によって光源環境が変わるような画像に対しては，単純な閾値処理より良い結果が得られる．
        img is NoneならNoneを、変換に成功すれば閾値処理された2値画像を返す。

        Parameters
        ----------
        img : OpenCV型
            変換前の画像データ
        gaussian : True or False
            ガウシアンフィルタを適応するか選択できる。
        method
            小領域中での閾値の計算方法
            ・cv2.ADAPTIVE_THRESH_MEAN_C : 近傍領域の中央値を閾値とする。
            ・cv2.ADAPTIVE_THRESH_GAUSSIAN_C : 近傍領域の重み付け平均値を閾値とする。
                                               重みの値はGaussian分布になるように計算。
        Type
            閾値の処理方法
            ・cv2.THRESH_BINARY
            ・cv2.THRESH_BINARY_INV
        block_size : int
            閾値計算に使用する近傍領域のサイズ。
            'ただし1より大きい奇数でなければならない。'
        C : int
            計算された閾値から引く定数。
        """
        if img is None:  # 画像の元データが存在しない場合
            return None
        # 画像のチャンネル数が2より大きい場合
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gaussian:
            img = cv2.GaussianBlur(img, (5, 5), 0)

        img = cv2.adaptiveThreshold(img, 255, method, Type, block_size, C)
        #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        #cv2.imshow('AdaptiveThresh', img)
        return img

    def TwoThreshold(self, img, gaussian=False, LowerThreshold=0, UpperThreshold=128, PickupColor=4, Type=cv2.THRESH_BINARY):
        """
        上側と下側の2つの閾値で2値化を行う。
        二値化には大局的閾値処理を用いる。

        Parameters
        ----------
        img : OpenCV型
            変換前の画像データ
        gaussian : True or False
            ガウシアンフィルタを適応するか選択できる。
        LowerThreshold : int
            下側の閾値
            範囲：0～127
        UpperThreshold : int
            上側の閾値
            範囲：128~256
        PickupColor : int
            抽出したい色を指定する。
            デフォルトは黒
            0: 赤
            1: 緑
            2: 青
            3: 白
            4: 黄色
        Type
            閾値の処理方法
            ・cv2.THRESH_BINARY
            ・cv2.THRESH_BINARY_INV
            ・cv2.THRESH_TRUNC
            ・cv2.THRESH_TOZERO
            ・cv2.THRESH_TOZERO_INV
        
        Returns
        -------
        result : int
            処理が成功したか確認するための返り値
            0: img画像が存在しなかった
            1: 抽出したい色の指定が不正
            2: 変換成功
        IMAGE_bw : OpenCV型
            変換後の画像データ
        """
        if img is None:  # 画像の元データが存在しない場合
            return 0, None
        r, g, b = cv2.split(img)

        # for Red
        IMAGE_R_bw= self.GlobalThreshold(r, gaussian, LowerThreshold, Type)
        IMAGE_R__ = self.GlobalThreshold(r, gaussian, UpperThreshold, Type)
        IMAGE_R__ = cv2.bitwise_not(IMAGE_R__)
        # for Green
        IMAGE_G_bw= self.GlobalThreshold(g, gaussian, LowerThreshold, Type)
        IMAGE_G__ = self.GlobalThreshold(g, gaussian, UpperThreshold, Type)
        IMAGE_G__ = cv2.bitwise_not(IMAGE_G__)
        # for Blue
        IMAGE_B_bw= self.GlobalThreshold(b, gaussian, LowerThreshold, Type)
        IMAGE_B__ = self.GlobalThreshold(b, gaussian, UpperThreshold, Type)
        IMAGE_B__ = cv2.bitwise_not(IMAGE_B__)
        
        if PickupColor == 0:
            IMAGE_bw = IMAGE_R_bw*IMAGE_G__*IMAGE_B__   # 画素毎の積を計算　⇒　赤色部分の抽出
        elif PickupColor == 1:
            IMAGE_bw = IMAGE_G_bw*IMAGE_B__*IMAGE_R__   # 画素毎の積を計算　⇒　緑色部分の抽出
        elif PickupColor == 2:
            IMAGE_bw = IMAGE_B_bw*IMAGE_R__*IMAGE_G__   # 画素毎の積を計算　⇒　青色部分の抽出
        elif PickupColor == 3:
            IMAGE_bw = IMAGE_R_bw*IMAGE_G_bw*IMAGE_B_bw # 画素毎の積を計算　⇒　白色部分の抽出
        elif PickupColor == 4:
            IMAGE_bw = IMAGE_R__*IMAGE_G__*IMAGE_B__    # 画素毎の積を計算　⇒　青色部分の抽出
        else:
            return 1, None
        
        return 2, IMAGE_bw

    def ExtractContours(self, img, RetrievalMode=cv2.RETR_LIST, ApproximateMode=cv2.CHAIN_APPROX_SIMPLE):
        """
        画像に含まれるオブジェクトの輪郭を抽出する関数。
        黒い背景（暗い色）から白い物体（明るい色）の輪郭を検出すると仮定。

        Parameters
        ----------
        img : OpenCV型
            変換前の画像データ
        RetrievalMode
            輪郭の階層情報
            cv2.RETR_LIST:  輪郭の親子関係を無視する。
                            親子関係が同等に扱われるので、単なる輪郭として解釈される。
            cv2.RETR_CCOMP: 2レベルの階層に分類する。
                            物体の外側の輪郭を階層1、物体内側の穴などの輪郭を階層2として分類。
            cv2.RETR_TREE:  全階層情報を保持する。
        ApproximateMode
            輪郭の近似方法
            cv2.CHAIN_APPROX_NONE: 中間点も保持する。
            cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない。
        """
        if img is None:  # 画像の元データが存在しない場合
            return None
        # 画像のチャンネル数が2より大きい場合
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img_2 = img.copy()
        img_2, contours, hierarchy = cv2.findContours(img_2, RetrievalMode, ApproximateMode)

        


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
        Connect = [
            [sg.Button('Conect to DOBOT', key='-Connect-'),
             sg.Button('Disconnect to DOBOT', key='-Disconnect-')],
            [sg.Button('WEB CAM on/off', key='-SetWebCam-'),
             sg.Button('Preview opened/closed', key='-Preview-'),],
        ]

        CurrentPose = [
            [sg.Button('Get Pose', size=(7, 1), key='-GetPose-')],
            [sg.Text('J1', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-JointPose1-'),
             sg.Text('X', size=(1, 1)),  sg.InputText('', size=(5, 1), disabled=True, key='-CoordinatePose_X-')],
            [sg.Text('J2', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-JointPose2-'),
             sg.Text('Y', size=(1, 1)),  sg.InputText('', size=(5, 1), disabled=True, key='-CoordinatePose_Y-')],
            [sg.Text('J3', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-JointPose3-'),
             sg.Text('Z', size=(1, 1)),  sg.InputText('', size=(5, 1), disabled=True, key='-CoordinatePose_Z-')],
            [sg.Text('J4', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-JointPose4-'),
             sg.Text('R', size=(1, 1)),  sg.InputText('', size=(5, 1), disabled=True, key='-CoordinatePose_R-')],
        ]
        
        SetPose = [
            [sg.Button('Set pose', size=(7, 1), key='-SetJointPose-'),
             sg.Button('Set pose', size=(7, 1), key='-SetCoordinatePose-'), ],
            [sg.Text('J1', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_1-'),
             sg.Text('X', size=(1, 1)), sg.InputText('', size=(5, 1), key='-CoordinatePoseInput_X-')],
            [sg.Text('J2', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_2-'),
             sg.Text('Y', size=(1, 1)), sg.InputText('', size=(5, 1), key='-CoordinatePoseInput_Y-')],
            [sg.Text('J3', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_3-'),
             sg.Text('Z', size=(1, 1)), sg.InputText('', size=(5, 1), key='-CoordinatePoseInput_Z-')],
            [sg.Text('J4', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_4-'),
             sg.Text('R', size=(1, 1)), sg.InputText('', size=(5, 1), key='-CoordinatePoseInput_R-')],
        ]
        
        WebCam = [
            [sg.Button('Snapshot', key='-Snapshot-'),
             sg.Button('Contours', key='-Contours-'),],
            [sg.Text('width', size=(4, 1)), sg.InputText('0', size=(5, 1), disabled=True, key='-IMAGE_width-'),
             sg.Text('height', size=(5, 1)), sg.InputText('0', size=(5, 1), disabled=True, key='-IMAGE_height-'), 
             sg.Text('channel', size=(5, 1)), sg.InputText('0', size=(5, 1), disabled=True, key='-IMAGE_channel-'),]
        ]

        ColorOfObject = [
            [sg.Radio('R', group_id='color', background_color='grey59', text_color='red', key='-color_R-'),
             sg.Radio('G', group_id='color', background_color='grey59', text_color='green', key='-color_G-'),
             sg.Radio('B', group_id='color', background_color='grey59', text_color='blue', key='-color_B-'),
             sg.Radio('W', group_id='color', background_color='grey59', text_color='snow', key='-color_W-'),
             sg.Radio('Bk', group_id='color', default=True, background_color='grey59', text_color='grey1', key='-color_Bk-')],
        ]

        Threshold_Type = [
            [sg.Text('閾値の処理方法'),
             sg.InputCombo(('BINARY',
                            'BINARY_INV',
                            'TRUNC',
                            'TOZERO',
                            'TOZERO_INV', ), size=(12, 1), key='-Threshold_type-', readonly=True),],
            [sg.Checkbox('ガウシアンフィルタ', key='-Gaussian-'), ],
        ]

        # 画像から物体の輪郭を切り出す関数の設定部分_GUI
        # 輪郭のモードを指定する
        RetrievalMode = [
            [sg.Text('輪郭'),],
            [sg.InputCombo(('親子関係を無視する',
                            '2つの階層に分類する',
                            '全階層情報を保持する',), size=(20, 1), key='-ContourRetrievalMode-', readonly=True),]
        ]
        
        # 近似方法を指定する
        ApproximateMode = [
            [sg.Text('輪郭の近似方法')],
            [sg.InputCombo(('中間点を保持する',
                            '中間点を保持しない'), size=(18, 1), key='-ApproximateMode-', readonly=True),]
        ]

        Global_Threshold = [
            [sg.Radio('Global Threshold', group_id='threshold', key='-GlobalThreshold-'),],
            [sg.Text('Threshold', size=(7, 1)),
             sg.InputText('127', size=(4, 1), key='-threshold-')],
            [sg.Checkbox('大津', key='-OTSU-')],
            ]

        Adaptive_Threshold = [
            [sg.Radio('Adaptive Threshold', group_id='threshold', key='-AdaptiveThreshold-'),],
            [sg.InputCombo(('MEAN_C',
                            'GAUSSIAN_C'), size=(12, 1), key='-AdaptiveThreshold_type-', readonly=True)],
            [sg.Text('Block Size'), sg.Text('Constant')],
            [sg.InputText('11', size=(4, 1), key='-AdaptiveThreshold_BlockSize-'),
             sg.InputText('2', size=(2, 1), key='-AdaptiveThreshold_constant-')],
        ]

        TwoThreshold = [
            [sg.Radio('TwoThreshold', group_id='threshold', default=True, key='-Twohreshold-'),],
            [sg.Text('Lower', size=(4, 1)),
             sg.Slider(range=(0, 127), default_value=10, orientation='horizontal', size=(12, 12), key='-LowerThreshold-'),],
            [sg.Text('Upper', size=(4, 1)),
             sg.Slider(range=(128, 256), default_value=138, orientation='horizontal', size=(12, 12), key='-UpperThreshold-')]
        ]



        layout = [
            [sg.Col(Connect)],
            [sg.Col(CurrentPose), sg.Col(SetPose)],
            [sg.Col(WebCam), sg.Col(Threshold_Type)],
            [sg.Col(RetrievalMode), sg.Col(ApproximateMode),
             sg.Frame('Color of object', ColorOfObject, background_color='grey59'), ],
            [sg.Col(Global_Threshold), 
             sg.Col(Adaptive_Threshold), 
             sg.Col(TwoThreshold)],
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

        elif (event == '-Snapshot-') or (event == '-Contours-'):
            if self.cammera_connection == 0:
                sg.popup('WebCameraが接続されていません。', title='Camの接続')
                return
            
            result, img = self.Snapshot() # 静止画を撮影する。
            if result == 0:
                sg.popup('スナップショットを撮影できませんでした。', title='スナップショット')
                return

            sg.popup('スナップショットを撮影しました。', title='スナップショット')
            self.IMAGE_original = img.copy() # 撮影した画像を保存する
            cv2.imshow('Snapshot', img)  # スナップショットを表示する

            [y, x, z] = self.IMAGE_original.shape
            # 画面上にスナップショットした画像の縦横の長さおよびチャンネル数を表示する。
            self.Window['-IMAGE_width-'].update(str(x))
            self.Window['-IMAGE_height-'].update(str(y))
            self.Window['-IMAGE_channel-'].update(str(z))
            
            # 撮影した画像をRGBに変換する。
            response, img = self.AnyImage2RGB(img) 
            if response == 0:
                sg.popup('画像の変換ができませんでした。', title='変換')
                return

            # 二値化の変換タイプを選択する。
            Type = ThresholdTypeOption(values['-Threshold_type-'])
            if Type is None:
                sg.popup('存在しない変換方法です。', title='エラー')
                return
            # -------------------- #
            #   GlobalThreshold    #
            # -------------------- #
            if values['-GlobalThreshold-']:
                # ------------------ #
                #    大津の二値化     #
                # ------------------ #
                if values['-OTSU-']:
                    img = self.OtsuThreshold(img, values['-Gaussian-'])
                # 選択されていない　⇒　大局的閾値処理を行う場合
                else:
                        threshold = float(values['-threshold-'])
                        img = self.GlobalThreshold(img, values['-Gaussian-'], threshold, Type=Type)
                        if img is None:
                            sg.popup('変換画像が存在しません。', title='エラー')
                            return

            # -------------------- #
            #  AdaptiveThreshold   #
            # -------------------- #
            elif values['-AdaptiveThreshold-']:
                # 適応的処理のタイプを選択する。
                if values['-AdaptiveThreshold_type-'] == 'MEAN_C':
                    method = cv2.ADAPTIVE_THRESH_MEAN_C
                elif values['-AdaptiveThreshold_type-'] == 'GAUSSIAN_C':
                    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                else:
                    sg.popup('存在しない処理方法です。', title='エラー')
                    
                # 処理方法が適切か判定
                if (values['-Threshold_type-'] == 'TRUNC') or \
                   (values['-Threshold_type-'] == 'TOZERO') or \
                   (values['-Threshold_type-'] == 'TOZERO_INV'):
                    sg.popup('そのTresholdの処理方法は使うことができません。', title='エラー')
                    return

                # BlockSizeが奇数か判定
                block_size = int(values['-AdaptiveThreshold_BlockSize-'])
                if block_size % 2 == 0:
                    sg.popup('block sizeが奇数ではありません。', title='エラー')
                    return

                const = int(values['-AdaptiveThreshold_constant-'])
                img = self.AdaptiveThreshold(img, values['-Gaussian-'], method, Type, block_size=block_size, C=const)
                
            # -------------------- #
            #     TwoThreshold     #
            # -------------------- #
            elif values['-Twohreshold-']:
                LowerThresh = int(values['-LowerThreshold-']) # 下側の閾値
                UpperThresh = int(values['-UpperThreshold-']) # 上側の閾値
                # 抽出したい色を選択
                if values['-color_R-']: color=0
                elif values['-color_G-']: color=1
                elif values['-color_B-']: color=2
                elif values['-color_W-']: color=3
                elif values['-color_Bk-']: color=4

                result, img = self.TwoThreshold(img, values['-Gaussian-'], LowerThresh, UpperThresh, color, Type)
                # エラー判定
                if result == 0:
                    sg.popup('変換前の画像が存在しませんでした。', title='エラー')
                    return
                elif result == 1:
                    sg.popup('抽出したい色の指定が不正です。', title='エラー')
                    return
            
            self.IMAGE_original2 = img.copy() # 変換した画像を保存する。
            cv2.imshow('Convert', img)
            return

    
    def main(self):
        return sg.Window('Dobot', self.layout, default_element_size=(40, 1), background_color='grey90')

    def loop(self):
        while True:
            event, values = self.Window.Read(timeout=10)
            if event is 'Quit':
                break
            if event != '__TIMEOUT__':
                self.Event(event, values)
            

def ThresholdTypeOption(Type_name):
    Type = None
    if Type_name == 'BINARY':       Type = cv2.THRESH_BINARY
    elif Type_name == 'BINARY_INV': Type = cv2.THRESH_BINARY_INV
    elif Type_name == 'TRUNC':      Type = cv2.THRESH_TRUNC
    elif Type_name == 'TOZERO':     Type = cv2.THRESH_TOZERO
    elif Type_name == 'TOZERO_INV': Type = cv2.THRESH_TOZERO_INV
    
    return Type


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

