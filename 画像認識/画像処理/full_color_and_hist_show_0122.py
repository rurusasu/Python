from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2, PySimpleGUI as sg
import numpy as np

Cammera_num = 0
Image_height = 120
Image_width  = 160
ticks = [0, 42, 84, 127, 169, 211, 255]
dic = {'rgb': [None, None],
       'glay': [None, None],
       'xyz': [None, None],
       'hsv': [None, None],
       'hls': [None, None],
       'YCrCb':[None, None],
       'luv': [None, None],
       'lab': [None, None],
       'yuv': [None, None]}


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

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


layout = [[sg.Image(filename='', size=(Image_width, Image_height), key='rgb'), sg.Canvas(size=(Image_width, Image_height), key='rgb_CANV'),
           sg.Image(filename='', size=(Image_width, Image_height), key='glay'), sg.Canvas(size=(Image_width, Image_height), key='glay_CANV'),
           sg.Image(filename='', size=(Image_width, Image_height), key='xyz'), sg.Canvas(size=(Image_width, Image_height), key='xyz_CANV'),],
          [sg.Image(filename='', size=(Image_width, Image_height), key='hsv'), sg.Canvas(size=(Image_width, Image_height), key='hsv_CANV'),
           sg.Image(filename='', size=(Image_width, Image_height), key='hls'), sg.Canvas(size=(Image_width, Image_height), key='hls_CANV'),
           sg.Image(filename='', size=(Image_width, Image_height), key='YCrCb'), sg.Canvas(size=(Image_width, Image_height), key='YCrCb_CANV'),],
          [sg.Image(filename='', size=(Image_width, Image_height), key='luv'), sg.Canvas(size=(Image_width, Image_height), key='luv_CANV'),
           sg.Image(filename='', size=(Image_width, Image_height), key='lab'), sg.Canvas(size=(Image_width, Image_height), key='lab_CANV'),
           sg.Image(filename='', size=(Image_width, Image_height), key='yuv'), sg.Canvas(size=(Image_width, Image_height), key='yuv_CANV'),],]

window = sg.Window('full Color Show', layout, location=(800, 400))
cap = cv2.VideoCapture(Cammera_num)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, Image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Image_height)

while True:
    event, values = window.read(timeout=20, timeout_key='timeout')
    if event is None: break
    ret, frame = cap.read()
    if not cap.isOpened(): break
    
    for key, value in dic.items():
        CANVASname = key + '_CANV'
        src = frame.copy()
        # if文で判定
        if key in 'rgb': dst = src
        elif key in 'glay':dst = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        elif key in 'hsv': dst = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
        elif key in 'hls': dst = cv2.cvtColor(src, cv2.COLOR_RGB2HLS)
        elif key in 'xyz': dst = cv2.cvtColor(src, cv2.COLOR_RGB2XYZ)
        elif key in 'YCrCb': dst = cv2.cvtColor(src, cv2.COLOR_RGB2YCrCb)
        elif key in 'luv': dst = cv2.cvtColor(src, cv2.COLOR_RGB2LUV)
        elif key in 'lab': dst = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        elif key in 'yuv': dst = cv2.cvtColor(src, cv2.COLOR_RGB2YUV)
        
        value[0] = dst
        canvas_elem = window[CANVASname]
        canvas = canvas_elem.TKCanvas
        fig, ax = plt.subplots(figsize=(2, 2))
        ax = Image_hist(value[0], ax, ticks)
        
        if value[1]: delete_figure_agg(value[1])
        value[1] = draw_figure(canvas, fig)

        window[key].update(data=cv2.imencode('.png', value[0])[1].tobytes())
        value[1].draw()
        
        dic[key] = value
