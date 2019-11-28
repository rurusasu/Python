# cording : utf-8
import PySimpleGUI as sg

class Window:
    def __init__(self):
        self.layout = self.Layout()

    def Layout(self):
        layout = [[sg.Text('Window 1')],
                  [sg.Input('')],
                  [sg.Button('Read')]]
        return layout

    def main(self):
        return sg.Window('My new window', self.layout, location=(800, 500), return_keyboard_events=True)


if __name__ == "__main__":
    window = Window()
    window2 = Window()

    window = window.main()
    window2 = window2.main()
    while True:
        #------------------------------------
        # Window1
        #------------------------------------            
        event, values = window.read(timeout=0)
        if event is None:
            break
        elif event != '__TIMEOUT__':
            print(event, values)

        #------------------------------------
        # Window2
        #------------------------------------
        event, values = window2.read(timeout=0)
        if event is None:
            break
        elif event != '__TIMEOUT__': # readボタンを押すと出力をprint
            print(event, values)
