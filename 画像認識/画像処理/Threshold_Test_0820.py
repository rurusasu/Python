# from matplotlib import pyplot as plt
import datetime as dt
import os
import cv2
import PySimpleGUI as sg
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lib.utils.base_utils import *
from lib.utils.binarization import *

Cammera_num = 1
src = np.array([])
ret = None
fig_agg = None  # 画像のヒストグラムを表示する用の変数
Image_height = 256  # 画面上に表示する画像の高さ
Image_width = 256  # 画面上に表示する画像の幅
ticks = [0, 40, 80, 120, 160, 200, 240]
binary_color_num = {"r": 0, "g": 1, "b": 2, "w": 3, "bk": 4}


class make_bin_img(object):
    img_elem = {
        "orig": np.array([]),
        "gray": np.array([]),
        "r": np.array([]),
        "g": np.array([]),
        "b": np.array([]),
    }
    bin_elem = {
        "r": np.array([]),
        "g": np.array([]),
        "b": np.array([]),
        "w": np.array([]),
        "bk": np.array([]),
    }
    bin_elem_rev = {
        "r_rev": np.array([]),
        "g_rev": np.array([]),
        "b_rev": np.array([]),
    }
    lower_thred = 0
    upper_thred = 0

    def __init__(self, src: np.ndarray, threshold: Type[Union[int, tuple]]) -> None:
        self.img_elem["orig"] = src
        if len(src.shape) == 1:
            self.img_elem["gray"] = src
        elif len(src.shape) == 3:
            self.img_elem["gray"] = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            [self.img_elem["r"], self.img_elem["g"], self.img_elem["b"]] = Disassembly(
                src
            )

        # 1つの閾値しか入力されていない場合
        if type(threshold) == int and (0 <= threshold <= 255):
            self.lower_thred = threshold
        # 複数の閾値が設定されている場合
        elif type(threshold) == tuple and (0 <= min(threshold), max(threshold) <= 255):
            self.lower_thred = min(threshold)
            self.upper_thred = max(threshold)
        else:
            raise ValueError("Input value is invalid.")

    def binarize(self, binary_type: str):
        # 2つの閾値を用いる場合とそうでない処理で条件分岐
        # なぜなら、2つの閾値を用いる場合は最大2個、そうでない処理の場合は最大3個の独立した閾値を持つため。
        if values["-Binary_Type-"] == "Two_Thresh":
            self._two_thresh_binalize(
                LowerThreshold=self.lower_thred,
                UpperThreshold=self.upper_thred,
            )
        else:
            for i, color in enumerate(self.img_elem):
                if color == "orig" or color == "gray":
                    continue

                if binary_type == "Global":
                    ret, dst = binalize(self.img_elem[color], self.lower_thred)
                elif binary_type == "Otsu":
                    ret, dst = otsu_binalize(self.img_elem[color])
                elif binary_type == "Adaptive":
                    dst = adaptive_binalize(self.img_elem[color])

                # bit 反転
                dst_rev = cv2.bitwise_not(dst)
                self.bin_elem[color], self.bin_elem_rev[f"{color}_rev"] = dst, dst_rev

            self._w_and_bk_binarize()

        return self.img_elem, self.bin_elem

    def _w_and_bk_binarize(self) -> None:
        # White color pickup
        w = self.bin_elem["r"] * self.bin_elem["g"] * self.bin_elem["b"]
        self.bin_elem["w"] = w

        # bk color pickup
        bk = (
            self.bin_elem_rev["r_rev"]
            * self.bin_elem_rev["g_rev"]
            * self.bin_elem_rev["b_rev"]
        )
        self.bin_elem["bk"] = bk

    def _two_thresh_binalize(
        self,
        LowerThreshold: int = 0,
        UpperThreshold: int = 128,
        Type=cv2.THRESH_BINARY,
    ) -> None:
        """2つの閾値を用いて画像を2値化処理する関数
        Args:
            LowerThreshold (int, optional): 低い閾値. Defaults to 0.
            UpperThreshold (int, optional): 高い閾値. Defaults to 128.
            Type ([type], optional): 2値化処理のタイプ. Defaults to cv2.THRESH_BINARY.
        """
        color_list = ["r", "g", "b"]

        for c in color_list:
            _, self.bin_elem[c] = binalize(self.img_elem[c], LowerThreshold, Type)
            _, tmp = binalize(self.img_elem["r"], UpperThreshold, Type)
            self.bin_elem_rev[f"{c}_rev"] = cv2.bitwise_not(tmp)

        # 画素毎の積を計算　⇒　赤色部分の抽出
        dst_r = (
            self.bin_elem["r"] * self.bin_elem_rev["g_rev"] * self.bin_elem_rev["b_rev"]
        )
        # 画素毎の積を計算　⇒　緑色部分の抽出
        dst_g = (
            self.bin_elem_rev["r_rev"] * self.bin_elem["g"] * self.bin_elem_rev["b_rev"]
        )
        # 画素毎の積を計算　⇒　青色部分の抽出
        dst_b = (
            self.bin_elem_rev["r_rev"] * self.bin_elem_rev["g_rev"] * self.bin_elem["b"]
        )
        # 画素毎の積を計算　⇒　白色部分の抽出
        self.bin_elem["w"] = (
            self.bin_elem["r"] * self.bin_elem["g"] * self.bin_elem["b"]
        )
        # 画素毎の積を計算　⇒　黒色部分の抽出
        self.bin_elem["bk"] = (
            self.bin_elem_rev["r_rev"]
            * self.bin_elem_rev["g_rev"]
            * self.bin_elem_rev["b_rev"]
        )

        # ほかの色も保存
        self.bin_elem["r"] = dst_r
        self.bin_elem["g"] = dst_g
        self.bin_elem["b"] = dst_b


# ------------------------------
# 空間フィルタリングを行う関数
# ------------------------------
def gamma_Image(src, gm):
    gamma = gm / 10
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
    LUT = np.arange(256, dtype="uint8").reshape(-1, 1)
    LUT = np.array([f(a, x).astype("uint8") for x in LUT])
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
        color = ["k"]
    elif len(img.shape) == 3:
        color = ["r", "g", "b"]
    for (i, col) in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist = np.sqrt(hist)
        ax.plot(hist, color=col)

    if ticks:
        ax.set_xticks(ticks)
    if thresh:
        if type(thresh) == tuple:
            for i in thresh:
                ax.vlines(i, 0, hist.max(), "blue", linestyles="dashed")
        else:
            ax.vlines(thresh, 0, hist.max(), "blue", linestyles="dashed")
    ax.set_title("histogram")
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
            new_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        maxCont = contours[0]

        for c in contours:
            if len(maxCont) < len(c):
                maxCont = c

        M = cv2.moments(maxCont)
    if int(M["m00"]) == 0:
        return None
    try:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
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
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    cv2.drawContours(src, large_contours, -1, color=(0, 255, 0), thickness=1)


# ---------------------------------------------
# Matplotlibグラフを画面に表示するための関数
# ---------------------------------------------
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close("all")


# ---------------------
# Layout
# ---------------------
# 画像タイプ
WebCam = [
    [
        sg.Radio(
            "Camera",
            "picture",
            size=(8, 1),
            background_color="grey63",
            key="-WebCam-",
            pad=(0, 0),
        ),
        sg.Radio(
            "Image",
            "picture",
            default=True,
            size=(8, 1),
            background_color="grey63",
            key="-Image-",
            pad=(0, 0),
        ),
        sg.Text("Color Space", background_color="grey63"),
        sg.InputCombo(
            (
                "RGB",
                "HSV",
            ),
            size=(6, 1),
            key="-Color_space-",
            readonly=True,
        ),
    ],
    [
        sg.Text(
            "Input Image",
            size=(12, 1),
            background_color="grey59",
            justification="left",
            pad=(0, 0),
        ),
        sg.Checkbox(
            "show threshold",
            default=False,
            size=(15, 1),
            background_color="grey59",
            key="-thresh_prev-",
        ),
        sg.Text("Binary Type", background_color="grey63"),
        sg.InputCombo(
            (
                "Global",
                "Otsu",
                "Adaptive",
                "Bradley",
                "Two_Thresh",
            ),
            default_value="Global",
            size=(10, 1),
            key="-Binary_Type-",
            readonly=True,
        ),
    ],
    [
        sg.Input(size=(30, 1), disabled=True),
        sg.FileBrowse(
            button_text="image file choice",
            change_submits=True,
            enable_events=True,
            disabled=True,
            key="-Image_path-",
        ),
    ],
    [
        sg.Input(size=(30, 1), disabled=True),
        sg.FolderBrowse(
            button_text="save", size=(11, 1), enable_events=True, key="-save_path-"
        ),
    ],
    [
        sg.Text(
            "Pick up color",
            size=(26, 1),
            background_color="grey59",
            justification="center",
            pad=(0, 0),
        ),
        sg.Text(
            "Calicurate of COG",
            size=(17, 1),
            background_color="grey63",
            justification="center",
            pad=(0, 0),
        ),
    ],
    [
        sg.Radio(
            "R",
            group_id="color",
            background_color="grey59",
            text_color="red",
            key="-color_r-",
            pad=(0, 0),
        ),
        sg.Radio(
            "G",
            group_id="color",
            background_color="grey59",
            text_color="green",
            key="-color_g-",
            pad=(0, 0),
        ),
        sg.Radio(
            "B",
            group_id="color",
            background_color="grey59",
            text_color="blue",
            key="-color_b-",
            pad=(0, 0),
        ),
        sg.Radio(
            "W",
            group_id="color",
            background_color="grey59",
            text_color="snow",
            key="-color_w-",
            pad=(0, 0),
        ),
        sg.Radio(
            "Bk",
            group_id="color",
            default=True,
            background_color="grey59",
            text_color="grey1",
            key="-color_bk-",
            pad=(0, 0),
        ),
        sg.InputCombo(
            (
                "なし",
                "Image",
                "輪郭をもとに計算",
            ),
            size=(17, 1),
            key="-CenterOfGravity-",
            readonly=True,
        ),
        sg.Button(button_text="Execution", key="-exe-"),
    ],
]

slider = [
    [
        sg.Text(
            "Lower",
        ),
        sg.Slider(
            (0, 255),
            default_value=,
            resolution=1,
            orientation="horizontal",
            size=(28, 5),
            key="-LowerThresh-",
        ),
    ],
    [
        sg.Text("Upper"),
        sg.Slider(
            (0, 255),
            default_value=126,
            resolution=1,
            orientation="horizontal",
            size=(28, 5),
            key="-UpperThresh-",
        ),
    ],
]

image = [
    [
        sg.Text(
            "Original Image",
            size=(26, 1),
            background_color="grey59",
            justification="center",
            pad=(0, 0),
        ),
        sg.Text(
            "Input Image",
            size=(26, 1),
            background_color="grey59",
            justification="center",
            pad=(0, 0),
        ),
    ],
    [
        sg.Image(filename="", size=(Image_width, Image_height), key="-Image_orig-"),
        sg.Image(filename="", size=(Image_width, Image_height), key="-Image_gray-"),
        sg.Image(filename="", size=(Image_width, Image_height), key="-Image_r-"),
        sg.Image(filename="", size=(Image_width, Image_height), key="-Image_g-"),
        sg.Image(filename="", size=(Image_width, Image_height), key="-Image_b-"),
    ],
    [
        sg.Image(filename="", size=(Image_width, Image_height), key="-Image_r_bin-"),
        sg.Image(filename="", size=(Image_width, Image_height), key="-Image_g_bin-"),
        sg.Image(filename="", size=(Image_width, Image_height), key="-Image_b_bin-"),
        sg.Image(filename="", size=(Image_width, Image_height), key="-Image_w_bin-"),
        sg.Image(filename="", size=(Image_width, Image_height), key="-Image_bk_bin-"),
    ],
]

canvas = [
    [sg.Canvas(size=(Image_width, Image_height), key="-CANVAS_1-")],
]

layout = [
    [
        sg.Col(WebCam, background_color="grey63"),
        sg.Col(slider, background_color="grey63"),
    ],
    [
        sg.Col(image, background_color="grey63"),
        sg.Col(canvas, background_color="grey63"),
    ],
]


window = sg.Window("Demo Application - OpenCV Integration", layout, location=(800, 400))


def __init__():
    """初期化用関数"""
    # Setup values
    frame = np.zeros((640, 480))
    # Setup the camera as a capture device
    cap = cv2.VideoCapture(Cammera_num)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Image_height)

    # グラフの体裁を整える
    # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams["xtick.direction"] = "in"
    # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.width"] = 1.0  # x軸主目盛り線の線幅
    plt.rcParams["ytick.major.width"] = 1.0  # y軸主目盛り線の線幅
    plt.rcParams["font.size"] = 11  # フォントの大きさ
    plt.rcParams["axes.linewidth"] = 1.0  # 軸の線幅edge linewidth。囲みの太さ


# __init__()

while True:  # The PSG "Event Loop"
    # get events for the window with 20ms max wait
    event, values = window.read(timeout=20, timeout_key="timeout")
    if event == None or event == "__TIMEOUT__":
        break

    elif event == "-exe-":
        # <<< 画像を読み出す <<<
        if values["-Image_path-"]:
            ImagePath = values["-Image_path-"]
            src = cv2.imread(ImagePath)
        else:
            sg.popup("画像ファイルがロードできませんでした。")
            continue

        # <<< 画像処理 <<<
        if src.size != 0:  # 出力する画像が存在する場合
            # 画像をリサイズ
            src = scale_box(src, Image_width, Image_height)

            mk_bin = make_bin_img(
                src, threshold=(values["-LowerThresh-"], values["-UpperThresh-"])
            )
            img_dict, dst_dict = mk_bin.binarize(values["-Binary_Type-"])

            # オリジナル画像の要素群を表示
            for i, color in enumerate(img_dict):
                window[f"-Image_{color}-"].update(
                    data=cv2.imencode(".png", img_dict[color])[1].tobytes()
                )  # Update image in window

            # 二値化処理後の画像群を表示
            for i, color in enumerate(dst_dict):
                window[f"-Image_{color}_bin-"].update(
                    data=cv2.imencode(".png", dst_dict[color])[1].tobytes()
                )  # Update image in window

            # ------------------------------------------------
            # 画面上に撮影した画像のヒストグラムを表示する
            # ------------------------------------------------
            canvas_elem = window["-CANVAS_1-"]
            canvas = canvas_elem.TKCanvas
            fig, ax = plt.subplots(figsize=(4, 3))
            if values["-thresh_prev-"]:
                ax = Image_hist(src, ax, ticks, ret)
            else:
                ax = Image_hist(src, ax, ticks)

            # グラフ右と上の軸を消す
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["top"].set_visible(False)

            if fig_agg:
                delete_figure_agg(fig_agg)

            fig_agg = draw_figure(canvas, fig)
            fig_agg.draw()

            if values["-CenterOfGravity-"] == "なし":
                pass
            elif values["-CenterOfGravity-"] == "Image":
                (Gx, Gy), dst = CenterOfGravity(dst, 0)
                cv2.drawMarker(
                    dst,
                    (Gx, Gy),
                    (0, 255, 0),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=15,
                )
            elif values["-CenterOfGravity-"] == "輪郭をもとに計算":
                (Gx, Gy), dst = CenterOfGravity(dst, 1)
                cv2.drawMarker(
                    dst,
                    (Gx, Gy),
                    (0, 255, 0),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=15,
                )

        # <<< 画像保存 <<<
        if values["-save_path-"]:
            # <<< ディレクトリ作成 <<<
            filename = (
                values["-Binary_Type-"]
                + "_"
                + str(values["-LowerThresh-"])
                + "_"
                + str(values["-UpperThresh-"])
                + "_"
            )
            # 現在時刻を取得
            dt_now = dt.datetime.now()
            # フォルダ名用にyyyymmddの文字列を取得する
            today = dt_now.strftime("%Y%m%d%H%M")
            filepth = os.path.join(values["-save_path-"], filename + today)
            # ディレクトリの存在判定
            makedir(filepth)

            # <<< オリジナル画像群を保存 <<<
            for i, color in enumerate(img_dict):
                img_name = color + "_" + filename
                save_img(img_dict[color], filepth=filepth, filename=img_name)

            # <<< 二値画像を保存 <<<
            for i, color in enumerate(dst_dict):
                img_name = color + "_bin_" + filename
                save_img(dst_dict[color], filepth=filepth, filename=img_name)
        else:
            sg.popup("画像の保存先が設定されていません。")

    # カメラを使う場合
    if values["-WebCam-"]:
        _, src = cap.read()
        if not cap.isOpened():
            break
    # 画像を使う場合
    elif values["-Image-"]:
        if window["-Image_path-"].Disabled:
            window["-Image_path-"].update(disabled=False)

    elif event == "-save-":
        print("file save")

    if type(src) == None:
        print("Not Found Error")
        break

        if values["-Color_space-"] == "HSV":
            src = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # src_1 = scale_box(frame, Image_width, Image_height)

    # for i in range(4):
    # window_num = '-Image_' + str(i) + '-'
    # window['-Image_0-'].update(data=cv2.imencode('.png', src)[1].tobytes()) # Update image in window
    # window['-Image_1-'].update(data=cv2.imencode('.png', src_1)[1].tobytes()) # Update image in window
    # window['-Image_2-'].update(data=cv2.imencode('.png', src_2)[1].tobytes()) # Update image in window
    # window['-Image_3-'].update(data=cv2.imencode('.png', src_3)[1].tobytes()) # Update image in window
    # window['image_2'].update(data=cv2.imencode('.png', dst)[1].tobytes()) # Update image in window
