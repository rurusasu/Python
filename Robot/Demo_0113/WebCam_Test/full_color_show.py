import cv2, PySimpleGUI as sg

Cammera_num = 0
Image_height = 120
Image_width  = 160

#def histgram_equalize(src, size):


layout = [[sg.Image(filename='', size=(Image_width, Image_height), key='-rgb-'),
           sg.Image(filename='', size=(Image_width, Image_height), key='-glay-'),
           sg.Image(filename='', size=(Image_width, Image_height), key='-hsv-'),],
          [sg.Image(filename='', size=(Image_width, Image_height), key='-hls-'),
           sg.Image(filename='', size=(Image_width, Image_height), key='-xyz-'), 
           sg.Image(filename='', size=(Image_width, Image_height), key='-YCrCb-'),],
          [sg.Image(filename='', size=(Image_width, Image_height), key='-luv-'),
           sg.Image(filename='', size=(Image_width, Image_height), key='-lab-'),
           sg.Image(filename='', size=(Image_width, Image_height), key='-yuv-'), ],]

window = sg.Window('full Color Show', layout, location=(800, 400))
cap = cv2.VideoCapture(Cammera_num)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, Image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Image_height)

while True:
    event, values = window.read(timeout=20, timeout_key='timeout')
    if event is None: break
    ret, frame = cap.read()
    if not cap.isOpened(): break
    
    # RGB
    RGB = frame.copy()
    #RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)
    # Glay
    Glay = frame.copy()
    Glay = cv2.cvtColor(Glay, cv2.COLOR_RGB2GRAY)
    # HSV
    HSV = frame.copy()
    HSV = cv2.cvtColor(HSV, cv2.COLOR_RGB2HSV)
    # HLS
    HLS = frame.copy()
    HLS = cv2.cvtColor(HLS, cv2.COLOR_RGB2HLS)
    # XYZ
    XYZ = frame.copy()
    XYZ = cv2.cvtColor(XYZ, cv2.COLOR_RGB2XYZ)
    # YCrCb
    YCrCb = frame.copy()
    YCrCb = cv2.cvtColor(YCrCb, cv2.COLOR_RGB2YCrCb)
    # LUV
    LUV = frame.copy()
    LUV = cv2.cvtColor(LUV, cv2.COLOR_RGB2LUV)
    # LAB
    LAB = frame.copy()
    LAB = cv2.cvtColor(LAB, cv2.COLOR_RGB2LAB)
    # YUV
    YUV = frame.copy()
    YUV = cv2.cvtColor(LAB, cv2.COLOR_RGB2YUV)
    

    window['-rgb-'].update(data=cv2.imencode('.png', RGB)[1].tobytes())
    window['-glay-'].update(data=cv2.imencode('.png', Glay)[1].tobytes())
    window['-hsv-'].update(data=cv2.imencode('.png', HSV)[1].tobytes())
    window['-hls-'].update(data=cv2.imencode('.png', HLS)[1].tobytes())
    window['-xyz-'].update(data=cv2.imencode('.png', XYZ)[1].tobytes())
    window['-YCrCb-'].update(data=cv2.imencode('.png', YCrCb)[1].tobytes())
    window['-luv-'].update(data=cv2.imencode('.png', LUV)[1].tobytes())
    window['-lab-'].update(data=cv2.imencode('.png', LAB)[1].tobytes())
    window['-yuv-'].update(data=cv2.imencode('.png', YUV)[1].tobytes())
