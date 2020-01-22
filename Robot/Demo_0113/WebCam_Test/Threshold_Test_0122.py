from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2, PySimpleGUI as sg
import numpy as np

Cammera_num = 1
fig_agg_1 = fig_agg_2 = fig_agg_3 = None     # 画像のヒストグラムを表示する用の変数
Image_height = 120 # 画面上に表示する画像の高さ
Image_width  = 160 # 画面上に表示する画像の幅


def binalize(src, threshold=127, Type=cv2.THRESH_BINARY):
    ret, img = cv2.threshold(src, threshold, 255, Type)
    return img

def adaptive_binalize(src, threshold=127, method=cv2.ADAPTIVE_THRESH_MEAN_C, Type=cv2.THRESH_BINARY, block_size=11, C=2):
    img = cv2.adaptiveThreshold(src, 255, method, Type, block_size, C)
    return img

def otsu_binalize(src):
    ret, img = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img

def gamma_Image(src, gm):
    gamma = gm/10
    img = LUT_curve(gamma_curve, gamma, src)
    return img
        
def LUT_curve(f, a, src):
    """
    Look Up Tableを LUT[input][0] = output という256行の配列として作る。
    例: LUT[0][0] = 0, LUT[127][0] = 160, LUT[255][0] = 255
    """
    LUT = np.arange(256, dtype='uint8').reshape(-1, 1)
    LUT = np.array([f(a, x).astype('uint8') for x in LUT])
    out_rgb_img = cv2.LUT(src, LUT)
    return out_rgb_img

def gamma_curve(gamma, x):
    y = 255 * pow(x / 255, 1.0 / gamma)
    return y

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

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')

def Image_hist(img, ax, ticks=None):
    """
    rgb_img と matplotlib.axes を受け取り、
    axes にRGBヒストグラムをplotして返す
    """
    if len(img.shape) == 2:
        color = ['k']
    elif len(img.shape) == 3:
        color = ['r', 'g', 'b']
    for (i, col) in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist = np.sqrt(hist)
        ax.plot(hist, color=col)

    if ticks:
        ax.set_xticks(ticks)
    ax.set_title('histogram')
    ax.set_xlim([0, 256])

    return ax

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

window = sg.Window('Demo Application - OpenCV Integration', 
                    [[sg.Image(filename='', key='image'),
                      sg.Canvas(size=(Image_width, Image_height), key='-CANVAS_1-')],
                     [sg.Image(filename='', key='image_2'),
                      sg.Canvas(size=(Image_width, Image_height), key='-CANVAS_2-')],
                     [sg.Image(filename='', key='image_3'),
                      sg.Canvas(size=(Image_width, Image_height), key='-CANVAS_3-')],], location=(800,400))
cap = cv2.VideoCapture(Cammera_num)       # Setup the camera as a capture device
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
while True:                     # The PSG "Event Loop"
    event, values = window.read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait
    if event is None:  break                                            # if user closed window, quit
    ret, frame = cap.read()
    src_1 = scale_box(frame, Image_width, Image_height)
    # bgr → glay
    src_2 = cv2.cvtColor(src_1, cv2.COLOR_BGR2GRAY)
    src_3 = src_4 = src_5 = src_6 = src_2.copy()
    # gamma変換
    src_3 = gamma_Image(src_3, 26)
    #------------------------------------------------
    # 画面上に撮影した画像のヒストグラムを表示する
    #------------------------------------------------
    canvas_elem_1 = window['-CANVAS_1-']
    canvas_1 = canvas_elem_1.TKCanvas
    canvas_elem_2 = window['-CANVAS_2-']
    canvas_2 = canvas_elem_2.TKCanvas
    canvas_elem_3 = window['-CANVAS_3-']
    canvas_3 = canvas_elem_3.TKCanvas
    ticks = [0, 42, 84, 127, 169, 211, 255]
    fig_1, ax_1 = plt.subplots(figsize=(3, 2))
    fig_2, ax_2 = plt.subplots(figsize=(3, 2))
    fig_3, ax_3 = plt.subplots(figsize=(3, 2))
    ax_1 = Image_hist(src_1, ax_1, ticks)
    ax_2 = Image_hist(src_2, ax_2, ticks)
    ax_3 = Image_hist(src_3, ax_3, ticks)

    if fig_agg_1: delete_figure_agg(fig_agg_1)
    if fig_agg_2: delete_figure_agg(fig_agg_2)
    if fig_agg_3: delete_figure_agg(fig_agg_3)

    fig_agg_1 = draw_figure(canvas_1, fig_1)
    fig_agg_2 = draw_figure(canvas_2, fig_2)
    fig_agg_3 = draw_figure(canvas_3, fig_3)
    fig_agg_1.draw()
    fig_agg_2.draw()
    fig_agg_3.draw()

    window['image'].update(data=cv2.imencode('.png', src_1)[1].tobytes()) # Update image in window
    window['image_2'].update(data=cv2.imencode('.png', src_2)[1].tobytes()) # Update image in window
    window['image_3'].update(data=cv2.imencode('.png', src_3)[1].tobytes()) # Update image in window
"""
Putting the comment at the bottom so that you can see that the code is indeed 7 lines long.  And, there is nothing
done out of the ordinary to make it 7 lines long.  There are no ; for example.  OK, so the if statement is on one line
but that's the only place that you would traditionally see one more line.  So, call it 8 if you want.
"""
