# cording: utf-8

import sys
import os
sys.path.append(os.getcwd())

import PySimpleGUI as sg
from Dobot import*
from common.csvIO import dataConv


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

def DataMake_click(org_FileName, lrn_FileName, tst_FileName, digit):
    lrn_FileName = __filePath__(lrn_FileName)
    tst_FileName = __filePath__(tst_FileName)
    #dataConv(org_FileName, lrn_FileName, col_range_end=3, digit=digit)
    #dataConv(org_FileName, tst_FileName, col_range_first=4, col_range_end=7, digit=digit)


# ----- Menu Definition ----- #
menu_def = [['File', ['Open', 'Save', 'Exit', 'Properties']],
            ['Edit', []],
            ['Help'],]

# ----- Column Definition ----- #
saveOgn = [
    [sg.Text('OriginalData')],
    [sg.Text('FileName'), 
     sg.InputText('data.csv', size=(13, 1), key='-Original-')],
    [sg.Button('SaveOriginal', key='-SaveOriginal-')],
]

saveVal = [
    [sg.Text('ValidationData')],
    [sg.Text('FileName'),
     sg.InputText('val.csv', size=(13, 1), key='-Validation-')],
    [sg.Button('SaveValidation', key='-SaveValidation-')],
]

dataConv = [
    [sg.Text('OrigiData')],
    [sg.InputText('data.csv', size=(13, 1)), sg.FileBrowse(key='-orgData-')],
    [sg.Text('LearnData')],
    [sg.InputText('learn.csv', size=(13, 1), key='-lrnData-')],
    [sg.Text('TestData'),
     sg.Text('小数点以下の桁数')],
    [sg.InputText('test.csv', size=(13, 1), key='-tstData-'),
     sg.InputText('2', size=(5, 1), key='-Digit-')],
    [sg.Button('DataMake', key='-DataMake-')]]

NeuralNet = [
    #[sg.Text('NuralNet')],
    [sg.Text('層の種類'), sg.Text('ユニット数')],
    [sg.InputCombo(('input', 'Dense'), size=(15, 1)), sg.InputText('50', size=(5, 1))],
    [sg.Text('重みの初期値')],
    [sg.InputCombo(('Xavier', 'He'), size=(15, 1))],
    [sg.Text('活性化関数')],
    [sg.InputCombo(('relu', 'sigmoid', 'liner'), size=(15, 1))],
    [sg.Text('損失関数')],
    [sg.InputCombo(('mean_squared_error'), size=(20, 1))],
    [sg.Text('評価関数')],
    [sg.InputCombo(('r2', 'rmse'), size=(15, 1))],
]

layout = [
    [sg.Text('Dobotを接続する')], 
    [sg.Button('Conect', key='-Connect-')],
    [sg.Frame('Save', [
        [sg.Column(saveOgn)],
        [sg.Column(saveVal)],
        ]),
     sg.Frame('NuralNet', NeuralNet)],
    [sg.Frame('DataConv', dataConv)],
    [sg.Quit()],
]

window = sg.Window('Dobot', layout, default_element_size=(40, 1))

# ボタンを押したときのイベントとボタンが返す値を代入
#event, values = window.Read()

CON_STR = None
while True:
    event, values = window.Read(timeout=10)
    if event is None or event == 'Quit':
        break
    if event is '-Connect-':
        CON_STR = Dobot()
        CON_STR.connect()
    elif event is '-SaveOriginal-':
        if CON_STR is None:
            print('Dobotに接続していません。')
        else:
            SaveOriginal_click(CON_STR, values['-Original-'])
    elif event is '-SaveValidation-':
        if CON_STR is None:
            print('Dobotに接続していません。')
        else:
            SaveValidation_click(CON_STR, values['-Validation-'])
    elif event is '-DataMake-':
        DataMake_click(values['-orgData-'], values['-lrnData-'], values['-tstData-'], values['-Digit-'])
    
