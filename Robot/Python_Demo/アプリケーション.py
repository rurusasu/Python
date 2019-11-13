# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

import PySimpleGUI as sg
import numpy as np

import DobotDllType as dType
from common.DobotFunction import initDobot, Coordinate_Operation, Coordinate_OneAction
from ctypes import cdll
from NuralNet_API import NuralNetApp

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
    # Dobot Connect
    state = dType.ConnectDobot(api, "", 115200)[0] # ConectDobot(const char* pointName, int baudrate)
    if (state != dType.DobotConnect.DobotConnect_NoError):
        print('Dobotに接続できませんでした。')
    
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


def SaveOriginal_click(CON_STR, file_name):
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
            Coordinate_Operation(api, file_name, 'y', 10)

            #-------------------------
            # 以下X軸方向の動作
            #-------------------------
            if j % 2 == 0:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Coordinate_Operation(api, file_name, 'x', 10)
            else:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Coordinate_Operation(api, file_name, 'x', -10)

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
        Coordinate_Operation(api, file_name, 'z', -0.5*i, initPOS)

        #-------------------------
        # 以下Y軸方向の動作
        #-------------------------
        for j in range(0, y_roop):
            Coordinate_Operation(api, file_name, 'x', 0.5)

            #-------------------------
            # 以下X軸方向の動作
            #-------------------------
            if j % 2 == 0:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Coordinate_Operation(api, file_name, 'y', 0.5)
            else:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Coordinate_Operation(api, file_name, 'y', -0.5)

    print('testデータ取得が終了しました。')

def DobotAct(x_pos, y_pos, z_pos):
    Coordinate_OneAction(api, x=x_pos, y=y_pos, z=z_pos)


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
    [sg.]
]


layout = [
    [sg.Text('Dobotを接続する')], 
    [sg.Button('Conect', key='-Connect-')],
    [sg.Frame('Save', 
        [[sg.Column(saveOrg)],
         [sg.Column(saveVal)],]),
    ],
    [sg.Frame('移動座標', inputPoint)],
    [sg.Quit()],
]

window = sg.Window('Dobot', layout, default_element_size=(40, 1))

# ボタンを押したときのイベントとボタンが返す値を代入
#event, values = window.Read()

#CON_STR = Dobot()
#NuralNet = NuralNetApp()
while True:
    NuralNet = NuralNetApp()
    event, values = window.Read(timeout=10)
    if event is 'Quit':
        break
    if event is '-Connect-':
        Connect_click()

    elif event is '-SaveOriginal-':
        if CON_STR is None:
            print('Dobotに接続していません。')
        else:
            SaveOriginal_click(CON_STR, values['-orgSave-'])
    elif event is '-SaveValidation-':
        if CON_STR is None:
            print('Dobotに接続していません。')
        else:
            SaveValidation_click(CON_STR, values['-valSave-'])
    elif event is '-ACT-':
        if CON_STR is None:
            print('Dobotに接続していません。')
        else:
            DobotAct(values['-x-'], values['-y-'], values['-z-'])

