# cording: utf-8

import sys, os
sys.path.append(os.getcwd())

import PySimpleGUI as sg
import numpy as np
from Dobot import*


def __filePath__(file_name):
    path = './data/' + str(file_name)
    return path

# ----- The callback function ----- #
def SaveOriginal_click(CON_STR, file_name):
    x_roop = 100
    y_roop = 100
    z_roop = 2

    file_name = __filePath__(file_name)
    print(file_name + 'にデータを保存します。')

    initPOS = CON_STR.getpose()
    #-----------------------------
    # 以下Z軸方向の動作
    #-----------------------------
    for i in range(0, z_roop):
        print('第' + str(i + 1) + 'ステップ目')
        CON_STR.Operation(file_name, 'z', -i, initPOS)

        #-------------------------
        # 以下Y軸方向の動作
        #-------------------------
        for j in range(0, y_roop):
            CON_STR.Operation(file_name, 'y')

            #-------------------------
            # 以下X軸方向の動作
            #-------------------------
            if j % 2 == 0:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    CON_STR.Operation(file_name, 'x')
            else:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    CON_STR.Operation(file_name, 'x', -1)

    print('データ取得が終了しました。')

def SaveValidation_click(CON_STR, file_name):
    x_roop = 100
    y_roop = 100
    z_roop = 2

    file_name = __filePath__(file_name)
    print(file_name + 'にデータを保存します。')

    initPOS = CON_STR.getpose()
    #-----------------------------
    # 以下Z軸方向の動作
    #-----------------------------
    for i in range(0, z_roop):
        print('第' + str(i + 1) + 'ステップ目')
        CON_STR.Operation(file_name, 'z', -0.5*i, initPOS)

        #-------------------------
        # 以下Y軸方向の動作
        #-------------------------
        for j in range(0, y_roop):
            CON_STR.Operation(file_name, 'x', 0.5)

            #-------------------------
            # 以下X軸方向の動作
            #-------------------------
            if j % 2 == 0:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    CON_STR.Operation(file_name, 'y', 0.5)
            else:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    CON_STR.Operation(file_name, 'y', -0.5)

    print('testデータ取得が終了しました。')

def DobotAct(CON_STR, x_pos, y_pos, z_pos):
    CON_STR._OneAction(x_pos, y_pos, z_pos)


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


layout = [
    [sg.Text('Dobotを接続する')], 
    [sg.Button('Conect', key='-Connect-')],
    [sg.Frame('Save', 
        [[sg.Column(saveOrg)],
         [sg.Column(saveVal)],
        ]),
     sg.Frame('移動座標', inputPoint),
    ],
    [sg.Quit()],
]

window = sg.Window('Dobot', layout, default_element_size=(40, 1))

# ボタンを押したときのイベントとボタンが返す値を代入
#event, values = window.Read()

CON_STR = Dobot()
while True:
    event, values = window.Read(timeout=10)
    if event is 'Quit':
        break
    if event is '-Connect-':
        CON_STR.connect()
    elif event is '-SaveOriginal-':
        if CON_STR is None:
            print('Dobotに接続していません。')
        else:
            SaveOriginal_click(CON_STR, values['-orgSavel-'])
    elif event is '-SaveValidation-':
        if CON_STR is None:
            print('Dobotに接続していません。')
        else:
            SaveValidation_click(CON_STR, values['-valSave-'])
    elif event is '-ACT-':
        if CON_STR is None:
            print('Dobotに接続していません。')
        else:
            DobotAct(CON_STR, values['-x-'], values['-y-'], values['-z-'])

