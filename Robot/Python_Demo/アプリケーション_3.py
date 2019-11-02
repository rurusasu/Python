# cording: utf-8

import sys
import os
sys.path.append(os.getcwd())

import PySimpleGUI as sg
from Dobot import*
from common.csvIO import dataConv
from nn import nn


def __filePath__(file_name):
    path = './data/' + str(file_name)
    return path

def dataLoad(filePath, dtype):
    if filePath != str:
        filePath = str(filePath)
    x = np.loadtxt(
            filePath, #読み込むファイル名(例"save_data.csv")
            dtype=dtype,     #データのtype
            delimiter=",",   #区切り文字の指定
            ndmin=2          #配列の最低次元
        )
    return x


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


def Training_click(orgPath, batch_size, epochs, feature=None, valPath=None ):
    orgLRN_Path = orgPath[0]
    orgTrg_Path = orgPath[1]

    # データの読み込み
    #訓練データの読み込み
    x = dataLoad(orgLRN_Path)
    t = dataLoad(orgTrg_Path)

    # Validationデータ
    if valPath != None:
        valRLN_Path = valPath[0]
        valTrg_Path = valPath[1]

        x_val = dataLoad(valLRN_Path)
        t_val = dataLoad(valTrg_Path)

        validation = (x_val, t_val)
    else:
        validation = None

    nn(x, t, batch_size, epochs, feature, validation)


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

NetCreate = [
    [sg.Text('層の種類'), sg.Text('ユニット数')],
    [sg.InputCombo(('input', 'Dense'), size=(15, 1)), sg.InputText('50', size=(5, 1))],
    [sg.Text('重みの初期値')],
    [sg.InputCombo(('Xavier', 'He'), size=(15, 1))],
    [sg.Text('活性化関数')],
    [sg.InputCombo(('relu', 'sigmoid', 'liner'), size=(15, 1))],
    [sg.Text('損失関数')],
    [sg.InputCombo(('mean_squared_error'), size=(20, 1))],
    [sg.Text('評価関数')],
    [sg.InputCombo(('r2', 'rmse'), size=(15, 1))],]


NuralNet = [
    [sg.Text('学習用データ')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-orgLRN-')],
    [sg.Text('学習用ラベル')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-orgTrg-')],
    [sg.Text('Validation用データ')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-valRLN-')],
    [sg.Text('Validation用ラベル')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-valTrg-')],
    [sg.Radio('標準化', 'RADIO1', size=(10, 1))],
    [sg.Radio('正規化', 'RADIO1', size=(10, 1))],
    [sg.Radio('両方', 'RADIO1', size=(10, 1))],
    [sg.Text('Batch Size'), sg.Text('epochs')],
    [sg.Input(size=(10, 1), key='-Batch-'), sg.Input(size=(10, 1), key='-epochs-')],
    [sg.Button('Training', key='-TrainingRUN-')],
    [sg.Text('テスト用データ')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-tstRLN-')],
    [sg.Text('テスト用ラベル')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-tstTrg-')],
    ]


layout = [
    [sg.Text('Dobotを接続する')], 
    [sg.Button('Conect', key='-Connect-')],
    [sg.Frame('Save', 
        [[sg.Column(saveOrg)],
         [sg.Column(saveVal)],
        ]),
     sg.Frame('NetCreate', NetCreate)],
    [sg.Frame('DataConv', dataConv),
     sg.Frame('NuralNet', NuralNet),],
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
            SaveOriginal_click(CON_STR, values['-orgSavel-'])
    elif event is '-SaveValidation-':
        if CON_STR is None:
            print('Dobotに接続していません。')
        else:
            SaveValidation_click(CON_STR, values['-valSave-'])
    elif event is '-DataMake-':
        DataMake_click(values['-orgData-'], values['-lrnData-'], values['-tstData-'], values['-Digit-'])
    # NuralNet
    elif event is '-TrainingRUN-':
        #Training_click((values['-orgLRN-'], values['-orgTrg-']), values['Batch'], values['epochs'], 
        print(event, values)