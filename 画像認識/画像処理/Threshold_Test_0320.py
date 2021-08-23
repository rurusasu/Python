#from matplotlib import pyplot as plt
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import PySimpleGUI as sg
import numpy as np
from matplotlib import pyplot as plt

Cammera_num = 1
frame = None
ret = None
fig_agg = None     # 画像のヒストグラムを表示する用の変数
Image_height = 120  # 画面上に表示する画像の高さ
Image_width = 160  # 画面上に表示する画像の幅
ticks = [0, 40, 80, 120, 160, 200, 240]


# ----------------------------------
# 画像を要素毎に分解
# ----------------------------------
def Disassembly(src):
    """
    画像を3つの要素に分解

    Parameter
    ---------
    src : OpenCV型
        分解前の画像

    Return
    ------
    dst : list
        要素のリスト
    """
    src_1, src_2, src_3 = cv2.split(src)  # Original → R,G,B(or H,S,V)
    dst = [src_1, src_2, src_3]
    return dst


# ----------------------------------
# 二値化方法
# ----------------------------------
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
    ret, dst = cv2.threshold(
        new_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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
    r, g, b = cv2.split(src)

    # for Red
    _, IMAGE_R_bw = binalize(r, LowerThreshold, Type)
    _, IMAGE_R__ = binalize(r, UpperThreshold, Type)
    IMAGE_R__ = cv2.bitwise_not(IMAGE_R__)
    # for Green
    _, IMAGE_G_bw = binalize(g, LowerThreshold, Type)
    _, IMAGE_G__ = binalize(g, UpperThreshold, Type)
    IMAGE_G__ = cv2.bitwise_not(IMAGE_G__)
    # for Blue
    _, IMAGE_B_bw = binalize(b, LowerThreshold, Type)
    _, IMAGE_B__ = binalize(b, UpperThreshold, Type)
    IMAGE_B__ = cv2.bitwise_not(IMAGE_B__)

    if PickupColor == 0:
        dst = IMAGE_R_bw*IMAGE_G__*IMAGE_B__   # 画素毎の積を計算　⇒　赤色部分の抽出
    elif PickupColor == 1:
        dst = IMAGE_G_bw*IMAGE_B__*IMAGE_R__   # 画素毎の積を計算　⇒　緑色部分の抽出
    elif PickupColor == 2:
        dst = IMAGE_B_bw*IMAGE_R__*IMAGE_G__   # 画素毎の積を計算　⇒　青色部分の抽出
    elif PickupColor == 3:
        dst = IMAGE_R_bw*IMAGE_G_bw*IMAGE_B_bw  # 画素毎の積を計算　⇒　白色部分の抽出
    elif PickupColor == 4:
        dst = IMAGE_R__*IMAGE_G__*IMAGE_B__    # 画素毎の積を計算　⇒　黒色部分の抽出

    ret_low, ret_up = LowerThreshold, UpperThreshold

    return (ret_low, ret_up), dst


# ------------------------------
# 空間フィルタリングを行う関数
# ------------------------------
def gamma_Image(src, gm):
    gamma = gm/10
    img = LUT_curve(gamma_curve, gamma, src)
    return img


# ----------------------------------------
# 空間フィルタリング用LTU(Look Up Table)
# ----------------------------------------
def LUT_curve(f, a, src):
    """
    Look Up Tableを LUT[input][0] = output という256行の配列として作る。
    例: LUT[0][0] = 0, LUT[127][0] = 160, LUT[255][0] = 255
    """
    LUT = np.arange(256, dtype='uint8').reshape(-1, 1)
    LUT = np.array([f(a, x).astype('uint8') for x in LUT])
    out_rgb_img = cv2.LUT(src, LUT)
    return out_rgb_img


# ----------------------------
# フィルタリング関数
# ----------------------------
def gamma_curve(gamma, x):
    y = 255 * pow(x / 255, 1.0 / gamma)
    return y


# ----------------------------
# ヒストグラムを計算する関数
# ----------------------------
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
        if type(thresh) is tuple:
            for i in thresh:
                ax.vlines(i, 0, hist.max(), "blue", linestyles='dashed')
        else:
            ax.vlines(thresh, 0, hist.max(), "blue", linestyles='dashed')
    ax.set_title('histogram')
    ax.set_xlim([0, 256])
    ax.set_ylim([0, 81])

    return ax


# ---------------------------------
# 重心位置を計算するための関数
# ---------------------------------
def CenterOfGravity(src, cal_Method=0):
    """
    オブジェクトの重心を計算する関数

    Parameter
    ---------
    org_img : OpenCV型
        重心計算方法用の画像
    contours : OpenCV型
        画像から抽出した輪郭情報
    cal_Method : int
        重心計算を行う方法を選択する
        0: 画像から重心を計算
        1: オブジェクトの輪郭から重心を計算

    Returns
    -------
    cx, cy : int
        オブジェクトの重心座標
    """
    new_img = src.copy()
    # 画像をもとに重心を求める場合
    if cal_Method == 0:
        M = cv2.moments(new_img, False)

    # 輪郭から重心を求める場合
    else:
        new_img, contours, hierarchy = cv2.findContours(
            new_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        maxCont = contours[0]

        for c in contours:
            if len(maxCont) < len(c):
                maxCont = c

        M = cv2.moments(maxCont)
    if int(M['m00']) == 0:
        return None
    try:
        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    except ZeroDivisionError:
        return None

    return (cx, cy), new_img


def drawing_edge(src, contours, min_area):
    """
    入力されたimgに抽出した輪郭線を描く関数

    Parameters
    ----------
    img : OpenCV型
        輪郭線を描く元データ
    contours : OpenCV型
        画像から抽出した輪郭情報
    min_area : int
        領域が占める面積の閾値を指定
    """
    large_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    cv2.drawContours(src, large_contours, -1, color=(0, 255, 0), thickness=1)


# ---------------------------------------------
# Matplotlibグラフを画面に表示するための関数
# ---------------------------------------------
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


# --------------------------------------
# アスペクト比を固定してリサイズする関数
# --------------------------------------
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


# ---------------------
# Layout
# ---------------------
# 画像タイプ
WebCam = [
    [sg.Radio('カメラ', 'picture', size=(5, 1), background_color='grey63', key='-WebCam-', pad=(0, 0)),
     sg.Radio('画像', 'picture', default=True, size=(3, 1),
              background_color='grey63', key='-Image-', pad=(0, 0)),
     sg.Text('Color Space', background_color='grey63'),
     sg.InputCombo(('RGB',
                    'HSV',), size=(6, 1), key='-Color_space-', readonly=True), ],
    [sg.Text('入力画像', size=(8, 1), background_color='grey59', justification='left', pad=(0, 0)),
     sg.Checkbox('閾値の表示', default=False, size=(10, 1),
                 background_color='grey59', key='-thresh_prev-'),
     sg.Text('Binary Type', background_color='grey63'),
     sg.InputCombo(('Global',
                    'Otsu',
                    'Adaptive',
                    'Bradley',
                    'Two_Thresh',), size=(6, 1), key='-Binary_Type-', readonly=True), ],
    [sg.Input(size=(30, 1), disabled=True),
     sg.FileBrowse(key='-Image_path-'), ],
    [sg.Text('抽出する色', size=(23, 1), background_color='grey59', justification='center', pad=(0, 0)),
     sg.Text('重心の計算方法', size=(17, 1), background_color='grey63', justification='center', pad=(0, 0))],
    [sg.Radio('R', group_id='color', background_color='grey59', text_color='red', key='-color_R-', pad=(0, 0)),
     sg.Radio('G', group_id='color', background_color='grey59',
              text_color='green', key='-color_G-', pad=(0, 0)),
     sg.Radio('B', group_id='color', background_color='grey59',
              text_color='blue', key='-color_B-', pad=(0, 0)),
     sg.Radio('W', group_id='color', background_color='grey59',
              text_color='snow', key='-color_W-', pad=(0, 0)),
     sg.Radio('Bk', group_id='color', default=True, background_color='grey59',
              text_color='grey1', key='-color_Bk-', pad=(0, 0)),
     sg.InputCombo(('なし',
                    'Image',
                    '輪郭をもとに計算', ), size=(17, 1), key='-CenterOfGravity-', readonly=True)],
]

slider = [
    [sg.Text('Lower', ),
     sg.Slider((0, 255), default_value=126, resolution=1, orientation='horizontal', size=(28, 5), key='-LowerThresh-')],
    [sg.Text('Upper'),
     sg.Slider((0, 255), default_value=126, resolution=1, orientation='horizontal', size=(28, 5), key='-UpperThresh-')],
]

image = [
    [sg.Image(filename='', size=(Image_width, Image_height), key='-Image_org-'),
     sg.Image(filename='', size=(Image_width, Image_height), key='-Image_0-'),
     sg.Image(filename='', size=(Image_width, Image_height), key='-Image_1-'),
     sg.Image(filename='', size=(Image_width, Image_height), key='-Image_2-'), ],
    [sg.Image(filename='', size=(Image_width, Image_height), key='-Im_Glay-'),
     sg.Image(filename='', size=(Image_width, Image_height), key='-Im_0_bin-'),
     sg.Image(filename='', size=(Image_width, Image_height), key='-Im_1_bin-'),
     sg.Image(filename='', size=(Image_width, Image_height), key='-Im_2_bin-'), ],
]

canvas = [
    [sg.Canvas(size=(Image_width, Image_height), key='-CANVAS_1-')],
]

layout = [
    [sg.Col(WebCam, background_color='grey63'), sg.Col(
        slider, background_color='grey63'), ],
    [sg.Col(image, background_color='grey63'), sg.Col(canvas, background_color='grey63')], ]


window = sg.Window('Demo Application - OpenCV Integration',
                   layout, location=(800, 400))

# Setup the camera as a capture device
cap = cv2.VideoCapture(Cammera_num)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, Image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Image_height)

# グラフの体裁を整える
# x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.direction'] = 'in'
# y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
plt.rcParams['font.size'] = 11  # フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ

while True:  # The PSG "Event Loop"
    # get events for the window with 20ms max wait
    event, values = window.read(timeout=20, timeout_key='timeout')
    if event is None:
        break

    elif event != '__TIMEOUT__':
        # カメラを使う場合
        if values['-WebCam-']:
            _, frame = cap.read()
            if not cap.isOpened():
                break
        # 画像を使う場合
        elif values['-Image-']:
            if values['-Image_path-']:
                ImagePath = values['-Image_path-']
                src = cv2.imread(ImagePath)
                frame = scale_box(src, Image_width, Image_height)

        if frame is not None:  # 出力する画像が存在する場合
            # if values['-Color_space-'] is 'RGB':
            #src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if values['-Color_space-'] is 'HSV':
                src = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #src_1 = scale_box(frame, Image_width, Image_height)
        else:
            continue

        src_glay = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        # Update image in window
        window['-Image_org-'].update(data=cv2.imencode('.png',
                                                       src)[1].tobytes())
        # Update image in window
        window['-Im_Glay-'].update(data=cv2.imencode('.png',
                                                     src_glay)[1].tobytes())
        dst_elem = Disassembly(src)
        # src_glay = cv2.cvtColor(src_bin, cv2.COLOR_RGB2GRAY)  # bgr → glay

        for i, src_bin in enumerate(dst_elem):  # 画像を3つの要素に分解
            window_num = '-Image_' + str(i) + '-'
            window_num_bin = '-Im_' + str(i) + '_bin-'
            if values['-Binary_Type-'] is 'Global':
                ret, dst = binalize(src_bin, values['-LowerThresh-'])
            elif values['-Binary_Type-'] is 'Otsu':
                ret, dst = otsu_binalize(src_bin)
            elif values['-Binary_Type-'] is 'Adaptive':
                dst = adaptive_binalize(src_bin)
                # elif values['-Binary_Type-'] is 'Bradley':
            elif values['-Binary_Type-'] is 'Two_Thresh':
                if values['-color_R-']:
                    pickup = 0
                elif values['-color_G-']:
                    pickup = 1
                elif values['-color_B-']:
                    pickup = 2
                elif values['-color_W-']:
                    pickup = 3
                elif values['-color_Bk-']:
                    pickup = 4
                ret, dst = Twothresh_binalize(
                    src_bin, LowerThreshold=values['-LowerThresh-'], UpperThreshold=values['-UpperThresh-'], PickupColor=pickup)
            window[window_num].update(data=cv2.imencode('.png', src_bin)[
                                      1].tobytes())  # Update image in window
            window[window_num_bin].update(data=cv2.imencode(
                '.png', dst)[1].tobytes())  # Update image in window

        # ------------------------------------------------
        # 画面上に撮影した画像のヒストグラムを表示する
        # ------------------------------------------------
        canvas_elem = window['-CANVAS_1-']
        canvas = canvas_elem.TKCanvas
        fig, ax = plt.subplots(figsize=(4, 3))
        if values['-thresh_prev-']:
            ax = Image_hist(src, ax, ticks, ret)
        else:
            ax = Image_hist(src, ax, ticks)

        # グラフ右と上の軸を消す
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        if fig_agg:
            delete_figure_agg(fig_agg)

        fig_agg = draw_figure(canvas, fig)
        fig_agg.draw()

        if values['-CenterOfGravity-'] is 'なし':
            pass
        elif values['-CenterOfGravity-'] is 'Image':
            (Gx, Gy), dst = CenterOfGravity(dst, 0)
            cv2.drawMarker(dst, (Gx, Gy), (0, 255, 0),
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=15)
        elif values['-CenterOfGravity-'] is '輪郭をもとに計算':
            (Gx, Gy), dst = CenterOfGravity(dst, 1)
            cv2.drawMarker(dst, (Gx, Gy), (0, 255, 0),
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=15)

        # for i in range(4):
            #window_num = '-Image_' + str(i) + '-'
        # window['-Image_0-'].update(data=cv2.imencode('.png', src)[1].tobytes()) # Update image in window
        # window['-Image_1-'].update(data=cv2.imencode('.png', src_1)[1].tobytes()) # Update image in window
        # window['-Image_2-'].update(data=cv2.imencode('.png', src_2)[1].tobytes()) # Update image in window
        # window['-Image_3-'].update(data=cv2.imencode('.png', src_3)[1].tobytes()) # Update image in window
        # window['image_2'].update(data=cv2.imencode('.png', dst)[1].tobytes()) # Update image in window
