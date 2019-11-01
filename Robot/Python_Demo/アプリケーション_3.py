import PySimpleGUI as sg

# ----- Menu Definition ----- #
menu_def = [['File', ['Open', 'Save', 'Exit', 'Properties']],
            ['Edit', []],
            ['Help'],]

# ----- Column Definition ----- #
saveOgn = [
    [sg.Text('OriginalData')],
    [sg.InputText('data.csv', size=(13, 1), key='-RobotAct-'), sg.FileBrowse()],
    [sg.Button('SaveOriginal')],
]

saveVal = [
    [sg.Text('ValidationData')],
    [sg.InputText('val.csv', size=(13, 1), key='-RobotAct-'), sg.FileBrowse()],
    [sg.Button('SaveValidation')],
]

dataConv = [
    [sg.Text('OrigiData')],
    [sg.InputText('data.csv', size=(13, 1)), sg.FileBrowse()],
    [sg.Text('LearnData')],
    [sg.InputText('learn.csv', size=(13, 1))],
    [sg.Text('TestData'),
     sg.Text('小数点以下の桁数')],
    [sg.InputText('test.csv', size=(13, 1)),
     sg.InputText('2', size=(5, 1))],
    [sg.Button('DataMake')]]

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
    [sg.Button('Conect')],
    [sg.Frame('Save', [
        [sg.Column(saveOgn)],
        [sg.Column(saveVal)],
        ]),
     sg.Frame('NuralNet', NeuralNet)],
    [sg.Frame('DataConv', dataConv)],
]

window = sg.Window('Dobot', layout, default_element_size=(40, 1))

# ボタンを押したときのイベントとボタンが返す値を代入
event, values = window.Read()

print(event, values)