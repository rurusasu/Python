import cv2, PySimpleGUI as sg

def binalize(src, threshold=127, Type=cv2.THRESH_BINARY):
    ret, img = cv2.threshold(src, threshold, 255, Type)
    return img

def adaptive_binalize(src, threshold=127, method=cv2.ADAPTIVE_THRESH_MEAN_C, Type=cv2.THRESH_BINARY, block_size=11, C=2):
    img = cv2.adaptiveThreshold(src, 255, method, Type, block_size, C)
    return img

def otsu_binalize(src):
    ret, img = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img

def Twothresh_binalize(src, LowerThreshold=0, UpperThreshold=128, PickupColor=4, Type=cv2.THRESH_BINARY):
    r, g, b = cv2.split(src)
    # for Red
    IMAGE_R_bw = binalize(r, LowerThreshold, Type)
    IMAGE_R__ = binalize(r, UpperThreshold, Type)
    IMAGE_R__ = cv2.bitwise_not(IMAGE_R__)
    # for Green
    IMAGE_G_bw = binalize(g, LowerThreshold, Type)
    IMAGE_G__ = binalize(g, UpperThreshold, Type)
    IMAGE_G__ = cv2.bitwise_not(IMAGE_G__)
    # for Blue
    IMAGE_B_bw = binalize(b, LowerThreshold, Type)
    IMAGE_B__ = binalize(b, UpperThreshold, Type)
    IMAGE_B__ = cv2.bitwise_not(IMAGE_B__)

    if PickupColor == 0:
        IMAGE_bw = IMAGE_R_bw*IMAGE_G__*IMAGE_B__   # 画素毎の積を計算　⇒　赤色部分の抽出
    elif PickupColor == 1:
        IMAGE_bw = IMAGE_G_bw*IMAGE_B__*IMAGE_R__   # 画素毎の積を計算　⇒　緑色部分の抽出
    elif PickupColor == 2:
        IMAGE_bw = IMAGE_B_bw*IMAGE_R__*IMAGE_G__   # 画素毎の積を計算　⇒　青色部分の抽出
    elif PickupColor == 3:
        IMAGE_bw = IMAGE_R_bw*IMAGE_G_bw*IMAGE_B_bw  # 画素毎の積を計算　⇒　白色部分の抽出
    elif PickupColor == 4:
        IMAGE_bw = IMAGE_R__*IMAGE_G__*IMAGE_B__    # 画素毎の積を計算　⇒　黒色部分の抽出
    else:
        return None

    return IMAGE_bw

window = sg.Window('Demo Application - OpenCV Integration', 
                    [[sg.Image(filename='', key='image'),
                      sg.Image(filename='', key='image_2'),
                      sg.Image(filename='', key='image_3')],
                     [sg.Image(filename='', key='image_4'),
                      sg.Image(filename='', key='image_5'),
                      sg.Image(filename='', key='image_6')],], location=(800,400))
cap = cv2.VideoCapture(1)       # Setup the camera as a capture device
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
while True:                     # The PSG "Event Loop"
    event, values = window.read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait
    if event is None:  break                                            # if user closed window, quit
    ret, frame = cap.read()
    src_1 = frame.copy()
    # bgr → glay
    src_2 = cv2.cvtColor(src_1, cv2.COLOR_BGR2GRAY)
    src_3 = src_4 = src_5 = src_6 = src_2.copy()
    # 大域的閾値選択
    src_3 = binalize(src_3)
    # 大津の二値化処理
    src_4 = otsu_binalize(src_4)
    # 適応的二値化処理
    src_5 = adaptive_binalize(src_5)
    # 2つの閾値を用いた処理
    src_6 = Twothresh_binalize(src_1)
    
    window['image'].update(data=cv2.imencode('.png', src_1)[1].tobytes()) # Update image in window
    window['image_2'].update(data=cv2.imencode('.png', src_2)[1].tobytes()) # Update image in window
    window['image_3'].update(data=cv2.imencode('.png', src_3)[1].tobytes()) # Update image in window
    window['image_4'].update(data=cv2.imencode('.png', src_4)[1].tobytes()) # Update image in window
    window['image_5'].update(data=cv2.imencode('.png', src_5)[1].tobytes()) # Update image in window
    window['image_6'].update(data=cv2.imencode('.png', src_6)[1].tobytes()) # Update image in window
"""
Putting the comment at the bottom so that you can see that the code is indeed 7 lines long.  And, there is nothing
done out of the ordinary to make it 7 lines long.  There are no ; for example.  OK, so the if statement is on one line
but that's the only place that you would traditionally see one more line.  So, call it 8 if you want.
"""
