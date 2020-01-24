#-*- coding:utf-8 -*-
import cv2, PySimpleGUI as sg
import numpy as np


Cammera_num = 0
frame = None
threshold_value = [80, 100, 120, 140, 160] # 閾値
Image_width, Image_height = 320, 240 # 表示画像サイズ


def scale_box(src, width, height):
    """
    アスペクト比を固定して、指定した大きさに収まるようリサイズする。

    Parameters
    ----------
    src : OpenCV型
        入力画像
    width : int
        変換後の画像幅
    height : int
        変換後の画像高さ
    
    Return
    ------
    dst : OpenCV型
    """
    scale = max(width / src.shape[1], height / src.shape[0])
    return cv2.resize(src, dsize=None, fx=scale, fy=scale)


layout = [[sg.Input(size=(30, 1)), sg.FileBrowse(key='-Image_path-'), 
           sg.Radio('カメラ', 'picture', size=(10, 1), key='-WebCam-'),
           sg.Radio('画像', 'picture', default=True, size=(10, 1), key='-Image-'), ],
          [sg.Image(filename='', key='image_1'),
           sg.Image(filename='', key='image_2'),
           sg.Image(filename='', key='image_3')],
          [sg.Image(filename='', key='image_4'),
           sg.Image(filename='', key='image_5'),
           sg.Image(filename='', key='image_6')],]

window = sg.Window('Demo Application - OpenCV Integration', layout, location=(800,400))
cap = cv2.VideoCapture(0)       # Setup the camera as a capture device
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
while True:                     # The PSG "Event Loop"
    event, values = window.read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait
    if event is None:  break                                            # if user closed window, quit
    # カメラを使う場合
    if values['-WebCam-']:
        ret, frame = cap.read()
        if not cap.isOpened(): break
    # 画像を使う場合
    elif values['-Image-']:
        if values['-Image_path-']:
            ImagePath = values['-Image_path-']
            frame = scale_box(cv2.imread(ImagePath), Image_width, Image_height)

    count = 2
    if frame is not None:
        src_rgb = frame.copy()
        # グレースケール変換
        src_gray = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY)
        # 画面に表示
        window['image_1'].update(data=cv2.imencode('.png', src_gray)[1].tobytes())

        for i in threshold_value:
            window_num = 'image_' + str(count)
            # 閾値処理    
            ret, src_bin = cv2.threshold(src_gray, i, 255, cv2.THRESH_BINARY)    
            # 画面に表示
            window[window_num].update(data=cv2.imencode('.png', src_bin)[1].tobytes())

            count += 1
    