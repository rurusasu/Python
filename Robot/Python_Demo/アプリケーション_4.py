# cording: utf-8

import sys, os
sys.path.append(os.getcwd())
import numpy as np
import PySimpleGUI as sg
from common.layers import Input, Dense
from common.sequential import Sequential
from common.functions import Datafeature
from common.callbacks import LearningVisualizationCallback
from common.csvIO import DataConv


def __filePath__(file_name):
    path = './data/' + str(file_name)
    return path

def dataLoad(filePath, dtype):
    if filePath != str:
        filePath = str(filePath)
    x = np.loadtxt(
        filePath,  # 読み込むファイル名(例"save_data.csv")
        dtype=dtype,  # データのtype
        delimiter=",",  # 区切り文字の指定
        ndmin=2  # 配列の最低次元
    )
    return x


# ----- The callback function ----- #
def DataMake_click(org_FileName, lrn_FileName, tst_FileName, digit):
    #lrn_FileName = __filePath__(lrn_FileName)
    #tst_FileName = __filePath__(tst_FileName)
    DataConv(org_FileName, lrn_FileName, col_range_end=3, digit=digit)
    DataConv(org_FileName, tst_FileName, col_range_first=4, col_range_end=7, digit=digit)


def LayerAdd_click(layerName, Node, weight=None, bias=None, activation=None):
    if Node != int:
            Node = int(Node)

    if layerName == 'input':
        model.add(Input(input_shape=Node))
    elif layerName == 'Dense':
        model.add(Dense(Node, activation, weight, bias))
    model.printParams()


def NetMake_click(loss, optimizer, metrics):
    metrics = [metrics]
    model.compile(loss, optimizer=optimizer, metrics=metrics)
    print('コンパイル完了')



def Training_click(orgPath, batch_size, epochs, feature=None, valPath=None):
    orgLRN_Path = orgPath[0]
    orgTrg_Path = orgPath[1]

    #---------------
    # 型変換
    #---------------
    if batch_size != int:
        batch_size = int(batch_size)
    if epochs != int:
        epochs = int(epochs)

    #---------------------
    # データの読み込み
    #---------------------
    x = dataLoad(orgLRN_Path, float)
    t = dataLoad(orgTrg_Path, float)
    #-------------------------------
    # DataFeature
    #-------------------------------
    if feature != None:
        x = Datafeature(x, feature)
        t = Datafeature(t, feature)
    #-----------------------------
    # Validationデータの読み込み
    #-----------------------------
    if valPath != None:
        valRLN_Path = valPath[0]
        valTrg_Path = valPath[1]
        x_val = dataLoad(valRLN_Path, float)
        t_val = dataLoad(valTrg_Path, float)
        #-------------------------------
        # DataFeature
        #-------------------------------
        if feature != None:
            x_val = Datafeature(x_val, feature)
            t_val = Datafeature(t_val, feature)

        validation = (x_val, t_val)
    else:
        validation = None

    # 学習曲線を可視化するコールバックを用意する
    higher_better_metrics = ['r2']
    visualize_cb = LearningVisualizationCallback(higher_better_metrics)
    callbacks = [
        visualize_cb,
    ]

    model.fit(x=x, t=t, batch_size=batch_size, epochs=epochs, validation=validation, callbacks=callbacks)


#def NetWorkTree():


# ----- Column Definition ----- #
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

NetMake = [
    [sg.Text('層の種類'), sg.Text('ユニット数')],
    [sg.InputCombo(('input', 'Dense'), size=(15, 1), key='-LayerName-'),
     sg.InputText('50', size=(5, 1), key='-Node-')],
    [sg.Text('重みの初期値')],
    [sg.InputCombo(('He', 'Xavier',), size=(15, 1), key='-weightInit-')],
    [sg.Text('閾値の初期値')],
    [sg.InputCombo(('ZEROS',), size=(15, 1), key='-biasInit-')],
    [sg.Text('活性化関数')],
    [sg.InputCombo(('relu', 'sigmoid', 'liner'), size=(15, 1), key='-activation-'),
     sg.Button('add', key='-LayerAdd-')],
    [sg.Text('損失関数')],
    [sg.InputCombo(('mean_squared_error',), size=(20, 1), key='-loss-')],
    [sg.Text('最適化')],
    [sg.InputCombo(('sgd',
                    'momentum_sgd',
                    'nag',
                    'ada_grad',
                    'rmsprop',
                    'ada_delta',
                    'adam',), size=(20, 1), key='-optimizer-')],
    [sg.Text('評価関数')],
    [sg.InputCombo(('r2', 'rmse'), size=(15, 1), key='-metrics-')],
    [sg.Button('NetMake', key='-NetMake-')]]


NetMakeTree = [
    [sg.TreeData]
]

NuralNet = [
    [sg.Text('学習用データ')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-orgLRN-')],
    [sg.Text('学習用ラベル')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-orgTrg-')],
    [sg.Text('Validation用データ')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-valRLN-')],
    [sg.Text('Validation用ラベル')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-valTrg-')],
    [sg.Radio('標準化', 'feature', size=(10, 1), key='-feature_0-'),
     sg.Radio('正規化', 'feature', size=(10, 1), key='-feature_1-'),
     sg.Radio('両方', 'feature', size=(10, 1),  key='-feature_2-')],
    [sg.Text('Batch Size'), sg.Text('epochs')],
    [sg.InputText('128', size=(10, 1), key='-Batch-'),
     sg.InputText('100', size=(10, 1), key='-epochs-')],
    [sg.Button('Training', key='-TrainingRUN-')],
    [sg.Text('テスト用データ')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-tstRLN-')],
    [sg.Text('テスト用ラベル')],
    [sg.Input(size=(30, 1)), sg.FileBrowse(key='-tstTrg-')],
]


layout = [
    [sg.Frame('NetMake', NetMake),
     sg.Frame('NuralNet', NuralNet), ],
    [sg.Frame('DataConv', dataConv), ],
    [sg.Quit()],
]

window = sg.Window('NeuralNet', layout, default_element_size=(40, 1))

# コンストラクタ
model = Sequential()

while True:
    event, values = window.Read(timeout=10)
    if event is None or event == 'Quit':
        break
    if event is '-DataMake-':
        DataMake_click(values['-orgData-'], values['-lrnData-'], values['-tstData-'], values['-Digit-'])
    # Networkの層を追加する。
    elif event is '-LayerAdd-':
        LayerAdd_click(layerName = values['-LayerName-'], 
                        Node = values['-Node-'], 
                        weight = values['-weightInit-'],
                        bias = values['-biasInit-'],
                        activation = values['-activation-'])
    elif event is '-NetMake-':
        NetMake_click(values['-loss-'], values['-optimizer-'], values['-metrics-'])
        #NetMake_click(values['-loss-'])
    # Trainig
    elif event is '-TrainingRUN-':
        #Training_click((values['-orgLRN-'], values['-orgTrg-']), values['Batch'], values['epochs'], feature, 
        #----------------------------
        # ラジオボタンの条件分岐
        #----------------------------
        if values['-feature_0-'] is True:
            feature = 0
        elif values['-feature_1-'] is True:
            feature = 1
        elif values['-feature_2-'] is True:
            feature = 2
        else:
            feature = None
        #----------------------------
        # Validationの条件分岐
        #----------------------------
        if (values['-valRLN-'] != '' and values['-valTrg-'] != ''):
            val = (values['-valRLN-'], values['-valTrg-'])
        else:
            val = None

        Training_click((values['-orgLRN-'], values['-orgTrg-']), values['-Batch-'], values['-epochs-'], feature, val)
        #print(event, values)

