from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2, PySimpleGUI as sg
import numpy as np

Cammera_num = 0
frame = None
ret = None
fig_agg_1 = fig_agg_2 = fig_agg_3 = None     # 画像のヒストグラムを表示する用の変数
Image_height = 240 # 画面上に表示する画像の高さ
Image_width  = 320 # 画面上に表示する画像の幅
ticks = [0, 42, 84, 127, 169, 211, 255]


#----------------------------------
# 二値化方法
#----------------------------------
def binalize(src, threshold=127, Type=cv2.THRESH_BINARY):
    """
    単純閾値処理を行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    threshold : int
        閾値

    Returns
    -------
    ret : int
        入力した閾値と同値

    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()
    ret, dst = cv2.threshold(new_img, threshold, 255, Type)
    return ret, dst

def otsu_binalize(src):
    """
    大津の閾値処理を行う関数

    Parameter
    ---------
    new_img : OpenCV型
        入力画像
    
    Returns
    -------
    ret : int
        計算した閾値
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()
    ret, dst = cv2.threshold(new_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ret, dst

def adaptive_binalize(src, method=cv2.ADAPTIVE_THRESH_MEAN_C, Type=cv2.THRESH_BINARY, block_size=11, C=2):
    """
    適応的閾値処理を行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    """
    new_img = src.copy()
    dst = cv2.adaptiveThreshold(new_img, 255, method, Type, block_size, C)
    return dst

def Twothresh_binalize(src, LowerThreshold=0, UpperThreshold=128, PickupColor=4, Type=cv2.THRESH_BINARY):
    dst = img = src.copy()
    dst[img < LowerThreshold] = 0
    dst[img > UpperThreshold] = 255
    ret_low, ret_up = LowerThreshold, UpperThreshold
    return (ret_low, ret_up), dst


#------------------------------
# 空間フィルタリングを行う関数
#------------------------------
def gamma_Image(src, gm):
    gamma = gm/10
    img = LUT_curve(gamma_curve, gamma, src)
    return img


#----------------------------------------
# 空間フィルタリング用LTU(Look Up Table)
#----------------------------------------
def LUT_curve(f, a, src):
    """
    Look Up Tableを LUT[input][0] = output という256行の配列として作る。
    例: LUT[0][0] = 0, LUT[127][0] = 160, LUT[255][0] = 255
    """
    LUT = np.arange(256, dtype='uint8').reshape(-1, 1)
    LUT = np.array([f(a, x).astype('uint8') for x in LUT])
    out_rgb_img = cv2.LUT(src, LUT)
    return out_rgb_img


#----------------------------
# フィルタリング関数
#----------------------------
def gamma_curve(gamma, x):
    y = 255 * pow(x / 255, 1.0 / gamma)
    return y


#----------------------------
# ヒストグラムを計算する関数
#----------------------------
def Image_hist(img, ax, ticks=None, thresh=None):
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
    if thresh:
        #ax.vlines(thresh, 0, hist.max(), "blue", linestyles='dashed')
        for i in thresh:
            ax.vlines(i, 0, hist.max(), "blue", linestyles='dashed')
    ax.set_title('histogram')
    ax.set_xlim([0, 256])

    return ax


#---------------------------------------------
# Matplotlibグラフを画面に表示するための関数
#---------------------------------------------
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


#--------------------------------------
# アスペクト比を固定してリサイズする関数
#--------------------------------------
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


#---------------------
# Layout
#---------------------
# 画像タイプ
WebCam = [
    [sg.Radio('カメラ', 'picture', size=(5, 1), background_color='grey63', key='-WebCam-', pad=(0, 0)),
     sg.Radio('画像', 'picture', default=True, size=(3, 1), background_color='grey63', key='-Image-', pad=(0, 0)),
     sg.Text('Binary Type', background_color='grey63'),
     sg.InputCombo(('Global',
                    'Otsu',
                    'Adaptive',
                    'Bradley',
                    'Two_Thresh'), size=(6, 1), key='-Binary_Type-', readonly=True),],
    [sg.Input(size=(30, 1), disabled=True), sg.FileBrowse(key='-Image_path-'), ],
    [sg.Image(filename='', size=(Image_width, Image_height), key='image')],
    [sg.Image(filename='', size=(Image_width, Image_height), key='image_2')]
]

slider = [
    [sg.Text('Lower', ),
     sg.Slider((0, 255), default_value=126, resolution=1, orientation='horizontal', size=(28, 5), key='-LowerThresh-')],
    [sg.Text('Upper'),
     sg.Slider((0, 255), default_value=126, resolution=1, orientation='horizontal', size=(28, 5), key='-UpperThresh-')],
    [sg.Canvas(size=(Image_width, Image_height), key='-CANVAS-')], ]

layout = [
    [sg.Col(WebCam, background_color='grey63'),
     sg.Col(slider, background_color='grey63')],]


window = sg.Window('Demo Application - OpenCV Integration', layout, location=(800,400))

cap = cv2.VideoCapture(Cammera_num)       # Setup the camera as a capture device
cap.set(cv2.CAP_PROP_FRAME_WIDTH, Image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Image_height)

while True:                     # The PSG "Event Loop"
    event, values = window.read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait
    if event is None: break
    # カメラを使う場合
    if values['-WebCam-']:
        _, frame = cap.read()
        if not cap.isOpened(): break
    # 画像を使う場合
    elif values['-Image-']:
        if values['-Image_path-']:
            ImagePath = values['-Image_path-']
            frame = scale_box(cv2.imread(ImagePath), Image_width, Image_height)
    
    if frame is not None:
        src_1 = scale_box(frame, Image_width, Image_height)
        src_2 = cv2.cvtColor(src_1, cv2.COLOR_BGR2GRAY)  # bgr → glay
    else: continue

    if values['-Binary_Type-'] is 'Global':
        ret, dst = binalize(src_2, values['-LowerThresh-'])
    elif values['-Binary_Type-'] is 'Otsu':
        ret, dst = otsu_binalize(src_2)
    elif values['-Binary_Type-'] is 'Adaptive':
        dst = adaptive_binalize(src_2)
        #elif values['-Binary_Type-'] is 'Bradley':
    elif values['-Binary_Type-'] is 'Two_Thresh':
        ret, dst = Twothresh_binalize(src_2, LowerThreshold=values['-LowerThresh-'], UpperThreshold=values['-UpperThresh-'])


    #------------------------------------------------
    # 画面上に撮影した画像のヒストグラムを表示する
    #------------------------------------------------
    canvas_elem_1 = window['-CANVAS-']
    canvas_1 = canvas_elem_1.TKCanvas
    fig_1, ax_1 = plt.subplots()
    ax_1 = Image_hist(src_2, ax_1, ticks, ret)

    if fig_agg_1: delete_figure_agg(fig_agg_1)

    fig_agg_1 = draw_figure(canvas_1, fig_1)
    fig_agg_1.draw()

    print(src_2)

    window['image'].update(data=cv2.imencode('.png', src_2)[1].tobytes()) # Update image in window
    window['image_2'].update(data=cv2.imencode('.png', dst)[1].tobytes()) # Update image in window
