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
    while True:
        window = Window()
        window = window.main()
        event, values = window.read(timeout=0)
        if event is None:
            break
        elif event != '__timeout__':
            print(event, values)