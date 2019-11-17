from pathlib import Path
from PIL import Image
import sys
import os
import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt


# フォルダーとファイルの画像のBase64バージョン。 PNGファイル（PySimpleGUI27では機能しない可能性があり、GIFと交換可能）
folder_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABnUlEQVQ4y8WSv2rUQRSFv7vZgJFFsQg2EkWb4AvEJ8hqKVilSmFn3iNvIAp21oIW9haihBRKiqwElMVsIJjNrprsOr/5dyzml3UhEQIWHhjmcpn7zblw4B9lJ8Xag9mlmQb3AJzX3tOX8Tngzg349q7t5xcfzpKGhOFHnjx+9qLTzW8wsmFTL2Gzk7Y2O/k9kCbtwUZbV+Zvo8Md3PALrjoiqsKSR9ljpAJpwOsNtlfXfRvoNU8Arr/NsVo0ry5z4dZN5hoGqEzYDChBOoKwS/vSq0XW3y5NAI/uN1cvLqzQur4MCpBGEEd1PQDfQ74HYR+LfeQOAOYAmgAmbly+dgfid5CHPIKqC74L8RDyGPIYy7+QQjFWa7ICsQ8SpB/IfcJSDVMAJUwJkYDMNOEPIBxA/gnuMyYPijXAI3lMse7FGnIKsIuqrxgRSeXOoYZUCI8pIKW/OHA7kD2YYcpAKgM5ABXk4qSsdJaDOMCsgTIYAlL5TQFTyUIZDmev0N/bnwqnylEBQS45UKnHx/lUlFvA3fo+jwR8ALb47/oNma38cuqiJ9AAAAAASUVORK5CYII='
file_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABU0lEQVQ4y52TzStEURiHn/ecc6XG54JSdlMkNhYWsiILS0lsJaUsLW2Mv8CfIDtr2VtbY4GUEvmIZnKbZsY977Uwt2HcyW1+dTZvt6fn9557BGB+aaNQKBR2ifkbgWR+cX13ubO1svz++niVTA1ArDHDg91UahHFsMxbKWycYsjze4muTsP64vT43v7hSf/A0FgdjQPQWAmco68nB+T+SFSqNUQgcIbN1bn8Z3RwvL22MAvcu8TACFgrpMVZ4aUYcn77BMDkxGgemAGOHIBXxRjBWZMKoCPA2h6qEUSRR2MF6GxUUMUaIUgBCNTnAcm3H2G5YQfgvccYIXAtDH7FoKq/AaqKlbrBj2trFVXfBPAea4SOIIsBeN9kkCwxsNkAqRWy7+B7Z00G3xVc2wZeMSI4S7sVYkSk5Z/4PyBWROqvox3A28PN2cjUwinQC9QyckKALxj4kv2auK0xAAAAAElFTkSuQmCC'


starting_path = sg.popup_get_folder('Folder to display')

if not starting_path:
    sys.exit(0)

treedata = sg.TreeData()


#---------------------------------------------
# ファイルをTreeに読み込む部分
#---------------------------------------------
def add_files_in_folder(parent, dirname):
    """
    TreeDataに値を格納するための関数
    """
    files = os.listdir(dirname)
    for f in files:
        fullname = os.path.join(dirname, f) # ファイル名を取得
        if os.path.isdir(fullname): # フォルダーの場合は、フォルダーを追加して再帰します
            treedata.Insert(parent, fullname, f, values=[], icon=folder_icon)
        else:

            treedata.Insert(parent, fullname, f, values=[
                            os.stat(fullname).st_size], icon=file_icon)


add_files_in_folder('', starting_path)


def get_filePaths_in_folder(dirname, extension='.jpg'):
    """
    ディレクトリ内のすべてのファイルパスを取得する関数

    Parameters
    ----------
    dirname : str
        親ディレクトリ名
    extention : str
        取得したいファイルの拡張子を指定する
        初期値 : None
        例 : extention = '.jpg'

    Return
    ------
    dir_list : PathObject
    file_list : PathObject
    """
    files = Path(dirname)
    # 内部ディレクトリが存在するか確認
    dir_list = list(files.glob('**'))
    string = '**/*' + str(extension)
    file_list = list(files.glob(string))
    
    return dir_list, file_list


def fig_Open(file_Path):
    img = []
    
    for path in file_Path:
        img.append(Image.open(path))
    return img

def make_dir(parent):
    files = Path(paent)
    if files:
        print('既にディレクトリが存在します。')
    else:
        files.mkdir(parents=True)




#---------------------------------------------
# 画像変換を行う部分
#---------------------------------------------
def grayscale(IMG_Data_list):
    """
    グレースケールに変換する関数

    Parameters
    ----------
    IMG_Data_list : list
        画像データのリスト

    Retern
    ------
    img : list
        変換後の画像のリスト
    """
    img=[]
    
    for i in IMG_Data_list:
        img.append(i.convert('L'))
    return img


def RGB(IMG_Data_list):
    """
    RGB画像に変換する関数

    Parameters
    ----------
    IMG_Data_list : list
        画像データのリスト

    Retern
    ------
    img : list
        変換後の画像のリスト
    """
    img=[]

    for i in IMG_Data_list:
        img.append(i.convert('RGB'))
    return img


def binarization(IMG_Data_list):
    """
    2値画像に変換する関数

    Parameters
    ----------
    IMG_Data_list : list
        画像データのリスト

    Retern
    ------
    img : list
        変換後の画像のリスト
    """
    img=[]
    table = [0 for x in range(256) if x<230]

    for i in IMG_Data_list:
        gray = i.convert('L') # グレースケールに変換
        conv = table * len(i.getbands())
        img.append(gray.point(conv)) #値が230以下は0になる


def brightness(IMG_Data_list, scale=1.0):
    """
    画像の明るさを変更する関数

    Parameters
    ----------
    IMG_Data_list : list
        画像データのリスト
    scale : float
        明るさの倍率
        1 < scale : 明るくなる
        0 <= scale < 1 : 暗くなる

    Retern
    ------
    img : list
        変換後の画像のリスト
    """
    img=[]
    table = [x*scale for x in range(256)]

    for i in IMG_Data_list:
        conv = table * len(i.getbands())
        img.append(i.point(conv))
    return img


def resize(IMG_Data_list, width, height, resample=Image.NEAREST):
    """
    画像をリサイズする関数

    Parameters
    ----------
    IMG_Data_list : list
        画像データのリスト
    width : int
        変換後の画像の幅をピクセル単位で指定する
    height : int
        変換後の画像の高さをピクセル単位で指定する
    resample : 
        リサンプリングする際に使われるフィルタを指定する
        Filters
        - NEAREST
        - BOX
        - BILINEAR
        - HAMMING
        - BICUBIC
        - LANCZOS

    Retern
    ------
    img : list
        変換後の画像のリスト
    """
    img=[]

    for i in IMG_Data_list:
        img.append(i.resize((width, height), resample))
    return img


def rotation(IMG_Data_list, angle, expand=True):
    """
    画像を回転する関数

    Parameters
    ----------
    IMG_Data_list : list
        画像データのリスト
    angle : 
        回転角度を数値指定する
    expand :
        True : 回転時に画像が大きくなってしまう場合に画像を拡張する
        False : 拡張しない

    Retern
    ------
    img : list
        変換後の画像のリスト
    """
    img=[]

    for i in IMG_Data_list:
        img.append(i.rotate(angle, expand))
    return img


def turning(IMG_Data_list, method):
    """
    画像を回転する関数

    Parameters
    ----------
    IMG_Data_list : list
        画像データのリスト
    angle : 
        回転角度を数値指定する
    expand :
        True : 回転時に画像が大きくなってしまう場合に画像を拡張する
        False : 拡張しない

    Retern
    ------
    img : list
        変換後の画像のリスト
    """

layout_resize = [
    [sg.Checkbox('画像のリサイズ', default=False)],
    [sg.Text('幅     '), sg.Text('高さ     '), sg.Text('リサンプリング')],
    [sg.Input(size=(5, 1), key='-width-'), 
     sg.Input(size=(5, 1), key='-height-'),
     sg.InputCombo(('NEAREST',
                    'BOX',
                    'BILINEAR',
                    'HAMMING',
                    'BICUBIC',
                    'LANCZOS',), size=(10, 1), key='-resample-')],
    ]

layout_rotation = [
    [sg.Checkbox('画像の回転', default=False)],
    [sg.Column([[sg.Text('回転角度（°）', size=(10, 1))],
                [sg.Input(size=(5, 1), key='-angle-')]]),
     sg.Column([[sg.Text('回転による画像の拡大')],
                [sg.InputCombo(('する',
                                'しない'), size=(10, 1), key='-expand-')]]),],
    ]

layout_brightness = [
    [sg.Checkbox('明るさを変更', default=False, key='-brightness-')],
    [sg.Text('上限倍率'), sg.Text('下限倍率')],
    [sg.Input(default_text='1.5', size=(5, 1), key='-brightness_Max-'),
     sg.Input(default_text='0.5', size=(5, 1), key='-brightness_Min-')]
    ]


layout = [[sg.Text('インポートしたい画像フォルダ')],
          [sg.Tree(data=treedata,
                   headings=['Size', ],
                   auto_size_columns=False,
                   num_rows=20,
                   col0_width=30,
                   key='-TREE-',
                   show_expanded=False,
                   enable_events=True),
          ],
          [sg.Text('画像の変換')],
          [sg.Radio('グレースケール', 'Convert', key='-gray-'),
           sg.Radio('RGB', 'Convert', key='-rgb-'),
           sg.Radio('なし', 'Convert', default=True,  key='-none-')],
          [sg.Frame('機能',layout_resize, size=(20, 10)),
           sg.Frame('回転', layout_rotation),
           sg.Frame('明るさ', layout_brightness),],
          [sg.Button('Convert', key='-Convert-')],
          [sg.Button('Ok'), sg.Button('Cancel')]]

window = sg.Window('Tree Element Test', layout, default_element_size=(40, 1))

img_list = None
while True:     # Event Loop
    event, values = window.read()
    if event in (None, 'Cancel'):
        break
    elif event in '-TREE-':
        dir_list, file_list=get_filePaths_in_folder(values['-TREE-'][0])
        #print(dir_list)
        #print(file_list)
        img_list=fig_Open(file_list)
        print(img_list)
    elif event is '-Convert-':
        #----------------------------
        # ラジオボタンの条件分岐
        #----------------------------
        if values['-gray-'] is True:
            # グレースケールが選択されたとき
            img_list = grayscale(img_list)
        elif values['-rgb-'] is True:
            # RGBが選択されたとき
            img_list = RGB(img_list)
        #----------------------------
        # 画像の明るさを変更する
        #----------------------------
        if values['-brightness-'] is True:
            img_list = brightness(img_list)
        for img in img_list:
            img = np.array(img)
        plt.imshow(img)
        plt.show()
    #print(event, values)
window.close()