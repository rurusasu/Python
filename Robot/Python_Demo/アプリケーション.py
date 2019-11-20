# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

import PySimpleGUI as sg
import numpy as np

import DobotDllType as dType
from common.DobotFunction import initDobot, Operation, OneAction
from ctypes import cdll
from NeuralNet_API import NeuralNetApp

api = cdll.LoadLibrary('DobotDll.dll')

CON_STR = {
            dType.DobotConnect.DobotConnect_NoError:  'DobotConnect_NoError',
            dType.DobotConnect.DobotConnect_NotFound: 'DobotConnect_NotFound',
            dType.DobotConnect.DobotConnect_Occupied: 'DobotConnect_Occupied'
        }

def __filePath__(file_name):
    path = './data/' + str(file_name)
    return path

# ----- The callback function ----- #
def Connect_click():
    """
    Dobotを接続する関数

    Returns
    -------
    0 : int
        Dobotが接続されていない場合
    1 : int
        Dobotが接続された場合
    """
    # Dobot Connect
    state = dType.ConnectDobot(api, "", 115200)[0] # ConectDobot(const char* pointName, int baudrate)
    if (state != dType.DobotConnect.DobotConnect_NoError):
        print('Dobotに接続できませんでした。')
        return 0
    #Clean Command Queued
    dType.SetQueuedCmdClear(api)

    #Async Motion Params Setting
    dType.SetHOMEParams(api, 150, -200, 100, 0, isQueued=1)

    #Async Home
    dType.SetHOMECmd(api, temp=0, isQueued=1)

    #キューに入っているコマンドを実行
    dType.SetQueuedCmdStartExec(api)

    #キューに入っているコマンドを停止
    dType.SetQueuedCmdStopExec(api)

    # Dobotの初期設定
    initDobot(api)
    return 1

def SaveOriginal_click(file_name):
    x_roop = 10
    y_roop = 20
    z_roop = 1

    file_name = __filePath__(file_name)
    print(file_name + 'にデータを保存します。')

    #initPOS = dType.GetPose(api)
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
            Operation(api, 'y', 10, file_name)

            #-------------------------
            # 以下X軸方向の動作
            #-------------------------
            if j % 2 == 0:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Operation(api, 'x', 10, file_name)
            else:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Operation(api, 'x', -10, file_name)

    print('データ取得が終了しました。')


def SaveValidation_click(file_name):
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
        Operation(api, 'z', -0.5*i, file_name, initPOS)

        #-------------------------
        # 以下Y軸方向の動作
        #-------------------------
        for j in range(0, y_roop):
            Operation(api, 'x', 0.5, file_name)

            #-------------------------
            # 以下X軸方向の動作
            #-------------------------
            if j % 2 == 0:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Operation(api, 'y', 0.5, file_name)
            else:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Operation(api, 'y', -0.5, file_name)

    print('testデータ取得が終了しました。')

def DobotAct(x_pos, y_pos, z_pos):
    OneAction(api, x=x_pos, y=y_pos, z=z_pos)


def ACT_JOGMode_click(J1_Angle, J2_Angle, J3_Angle, J4_Angle):
    #Operation(api, 'z', J1_Angle, mode=dType.PTPMode.PTPMOVLANGLEMode)
    if J1_Angle is '':
        J1_Angle = dType.GetPose(api)[4]
    if J2_Angle is '':
        J2_Angle = dType.GetPose(api)[5]
    if J3_Angle is '':
        J3_Angle = dType.GetPose(api)[6]
    if J4_Angle is '':
        J4_Angle = dType.GetPose(api)[7]
    
    J1_Angle = float(J1_Angle)
    J2_Angle = float(J2_Angle)
    J3_Angle = float(J3_Angle)
    J4_Angle = float(J4_Angle)

    OneAction(api, J1_Angle, J2_Angle, J3_Angle, J4_Angle, mode=dType.PTPMode.PTPMOVLANGLEMode)
    print(J1_Angle)
    print(J2_Angle)
    print(J3_Angle)
    print(J4_Angle)



# ----- Menu Definition ----- #
menu_def = [['File', ['Open', 'Save', 'Exit', 'Properties']],
            ['Edit', []],
            ['Help'],]

# ----- Column Definition ----- #
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


printAngle = [
    [sg.Text('J1'), sg.Input(size=(5, 1), key='-J1-')],
    [sg.Text('J2'), sg.Input(size=(5, 1), key='-J2-')],
    [sg.Text('J3'), sg.Input(size=(5, 1), key='-J3-')],
    [sg.Text('J4'), sg.Input(size=(5, 1), key='-J4-')],
    [sg.Button('ACT', key='-JOGAct-')]
]


layout = [
    [sg.Text('Dobotを接続する')], 
    [sg.Button('Conect', key='-Connect-')],
    [sg.Frame('Save', 
        [[sg.Column(saveOrg)],
         [sg.Column(saveVal)],]),
     sg.Frame('Angle', 
        [[sg.Column(printAngle)],]),
    ],
    [sg.Frame('移動座標', inputPoint), sg.Button('NeuralNetAPI', key='-NeuralNetAPI-')],
    [sg.Quit()],
]

window = sg.Window('Dobot', layout, default_element_size=(40, 1))

# ボタンを押したときのイベントとボタンが返す値を代入
#event, values = window.Read()

#CON_STR = Dobot()
NuralNet = None
# Dobotの接続確認用変数
connect_Check = 0 # 0:DisConnect, 1:Connect
while True:
    #NuralNet = NuralNetApp()
    event, values = window.Read(timeout=10)
    if event is 'Quit':
        break
    elif event is '-Connect-':
        connect_Check = Connect_click()
        continue
    elif event is '-SaveOriginal-':
        # 接続の確認
        if connect_Check is 0: 
            print ('Dobotが接続されていません。')
            continue
        # 動作
        SaveOriginal_click(values['-orgSave-'])
    elif event is '-SaveValidation-':
        # 接続の確認
        if connect_Check is 0:
            print('Dobotが接続されていません。')
            continue
        # 動作
        SaveValidation_click(values['-valSave-'])
    elif event is '-ACT-':
        # 接続の確認
        if connect_Check is 0:
            print('Dobotが接続されていません。')
            continue
        # 動作
        DobotAct(values['-x-'], values['-y-'], values['-z-'])
    elif event is '-JOGAct-':
        # 接続の確認
        if connect_Check is 0:
            print('Dobotが接続されていません。')
            continue
        # 動作
        ACT_JOGMode_click(values['-J1-'], values['-J2-'],
                            values['-J3-'], values['-J4-'])
    elif event is '-NeuralNetAPI-':
        # 初回動作時
        if NuralNet is None:
            NuralNet = NeuralNetApp()
            continue
        else:
            print('APIはすでに起動しています。')
            continue

