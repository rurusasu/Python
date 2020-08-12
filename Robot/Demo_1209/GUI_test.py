import PySimpleGUI as sg
import cv2

"""
    Demo of using OpenCV to show your webcam in a GUI window.
    This demo will run on tkinter, Qt, and Web(Remi).  The web version flickers at the moment though
    To exit, right click and choose exit. If on Qt, you'll have to kill the program as there are no right click menus
    in PySimpleGUIQt (yet).
"""

sg.change_look_and_feel('Black')

# define the window layout
layout_1 = [
    [sg.Image(filename='', key='-IMAGE-', tooltip='Right click for exit menu')], ]

layout_2 = [[sg.Button('WebCam Open/Close', key='WebCam')], [sg.Quit()]]

# create the window and show it without the plot
window_1 = sg.Window('Demo Application - OpenCV Integration', layout_1, location=(800, 400),
                     no_titlebar=True, grab_anywhere=True,
                     right_click_menu=['&Right', ['E&xit']], )  # if trying Qt, you will need to remove this right click menu

window_2 = sg.Window('CamPreview', layout_2)

cam_open = 0
# ---===--- Event LOOP Read and display frames, operate the GUI --- #
# Setup the OpenCV capture device (webcam)
cap = cv2.VideoCapture(0)
while True:
    event_1, values_1 = window_1.read(timeout=20)
    event_2, values_2 = window_2.read(timeout=20)
    if event_2 is 'Quit':
        break
    if event_2 == 'WebCam' and cam_open == 0:
        #while True:
            #event, values = window_1.read(timeout=20)
            if event_1 in ('Exit', None):
                break
            # Read image from capture device (camera)
            ret, frame = cap.read()
            # Convert the image to PNG Bytes
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            # Change the Image Element to show the new image
            window_1['-IMAGE-'].update(data=imgbytes)

window_1.close()
window_2.close()
