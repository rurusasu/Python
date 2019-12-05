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
        #--- エラーフラグ ---#
        self.connection = 0 # Connect:1, DisConnect:0
        self.DOBOT_err = 0  # Error occurred:1, No error:0
        self.Input_err = 0  # Error occurred:1, No error:0
        # GUIの初期化
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

    def SetCoordinatePose_click(self):
        """
        指定された作業座標系にアームの先端を移動させる関数
        デカルト座標系で移動

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
        elif self.CurrentPose is None:
            response = 2
            return response

        timeout(5)
        try:
            dType.SetPTPCmd(self.api, 
                            dType.PTPMode.PTPMOVJXYZMode, 
                            self.CurrentPose[0], 
                            self.CurrentPose[1],
                            self.CurrentPose[2],
                            self.CurrentPose[3],
                            self.queue_index)
        except TimeoutError:
            self.DOBOT_err = 1
            return response
        response = 1
        return response

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
        """
        CoordinatePose = [
            [sg.Text('X', size=(5, 1)), sg.Text('', size=(5, 1), key='-CoordinatePose_X-')],
            [sg.Text('Y', size=(5, 1)), sg.Text('', size=(5, 1), key='-CoordinatePose_Y-')],
            [sg.Text('Z', size=(5, 1)), sg.Text('', size=(5, 1), key='-CoordinatePose_Z-')],
            [sg.Text('R', size=(5, 1)), sg.Text('', size=(5, 1), key='-CoordinatePose_R-')],
        ]
        """
        SetJointPose = [
            [sg.Button('Set pose', key='-SetJointPose-')],
            [sg.Text('J1', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_1-')],
            [sg.Text('J2', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_2-')],
            [sg.Text('J3', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_3-')],
            [sg.Text('J4', size=(2, 1)), sg.InputText('', size=(5, 1), key='-JointPoseInput_4-')],
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
            #[sg.Button('Get Pose', size=(7, 1), key='-GetPose-')],
            #[sg.Col(JointPose, size=(40, 100)), sg.Col(CoordinatePose, size=(40, 100))],
            [sg.Col(CurrentPose), sg.Col(SetJointPose)],
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
                sg.popup('Dobotに接続できませんでした。', title='Dobotの接続')
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
                sg.popup('Dobotは接続されていません。', title='Dobotの接続')
                return
            elif result == 1:
                sg.popup('Dobotの接続を切断しました。', title='Dobotの接続')
                self.connection = 0
                return

        # Dobotの現在の姿勢を取得し表示する
        elif event == '-GetPose-':
            if self.connection == 0:
                sg.popup('Dobotは接続されていません。', title='Dobotの接続')
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
                sg.popup('Dobotは接続されていません。', title='Dobotの接続')
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
                """
                if event == '-Connect-':
                    result = self.Connect_click()
                    if result == 0:
                        sg.popup('Dobotに接続できませんでした。', title='Dobotの接続')
                        self.connection = 0
                        return
                    elif result == 1:
                        sg.popup('Dobotに接続しました。', title='Dobotの接続')
                        self.connection = 1
                        return
                    else:
                        sg.popup('Dobotに接続しています。', title='Dobotの接続')
                        return

                elif event == '-Disconnect-':
                    result = self.Disconnect_click()
                    if result == 0:
                        sg.popup('Dobotは接続されていません。', title='Dobotの接続')
                        return
                    elif result == 1:
                        sg.popup('Dobotの接続を切断しました。', title='Dobotの接続')
                        self.connection = 0
                        return

                elif event == '-GetPose-':
                    if self.connection == 0:
                        sg.popup('Dobotは接続されていません。', title='Dobotの接続')
                        return
                    else:
                        response, pose = self.GetPose_click()
                        if response == 0:
                            sg.popup('Dobotからの応答がありません。', title='Dobotの接続')
                            return
                        else:
                            print(values)
                            self.Window['-JointPose1-'].update(str(pose[4]))
                            #values['-JointPose1-'] = str(pose[4])
                            values['-JointPose2-'] = str(pose[5])
                            values['-JointPose3-'] = str(pose[6])
                            values['-JointPose4-'] = str(pose[7])
                            values['-CoordinatePose_X-'] = str(pose[0])
                            values['-CoordinatePose_Y-'] = str(pose[1])
                            values['-CoordinatePose_Z-'] = str(pose[2])
                            values['-CoordinatePose_R-'] = str(pose[3])
                #elif event == '-SetJointPose-':

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
            """




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

