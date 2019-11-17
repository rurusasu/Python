import sys
import os
import PySimpleGUI as sg


# フォルダーとファイルの画像のBase64バージョン。 PNGファイル（PySimpleGUI27では機能しない可能性があり、GIFと交換可能）
folder_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABnUlEQVQ4y8WSv2rUQRSFv7vZgJFFsQg2EkWb4AvEJ8hqKVilSmFn3iNvIAp21oIW9haihBRKiqwElMVsIJjNrprsOr/5dyzml3UhEQIWHhjmcpn7zblw4B9lJ8Xag9mlmQb3AJzX3tOX8Tngzg349q7t5xcfzpKGhOFHnjx+9qLTzW8wsmFTL2Gzk7Y2O/k9kCbtwUZbV+Zvo8Md3PALrjoiqsKSR9ljpAJpwOsNtlfXfRvoNU8Arr/NsVo0ry5z4dZN5hoGqEzYDChBOoKwS/vSq0XW3y5NAI/uN1cvLqzQur4MCpBGEEd1PQDfQ74HYR+LfeQOAOYAmgAmbly+dgfid5CHPIKqC74L8RDyGPIYy7+QQjFWa7ICsQ8SpB/IfcJSDVMAJUwJkYDMNOEPIBxA/gnuMyYPijXAI3lMse7FGnIKsIuqrxgRSeXOoYZUCI8pIKW/OHA7kD2YYcpAKgM5ABXk4qSsdJaDOMCsgTIYAlL5TQFTyUIZDmev0N/bnwqnylEBQS45UKnHx/lUlFvA3fo+jwR8ALb47/oNma38cuqiJ9AAAAAASUVORK5CYII='
file_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABU0lEQVQ4y52TzStEURiHn/ecc6XG54JSdlMkNhYWsiILS0lsJaUsLW2Mv8CfIDtr2VtbY4GUEvmIZnKbZsY977Uwt2HcyW1+dTZvt6fn9557BGB+aaNQKBR2ifkbgWR+cX13ubO1svz++niVTA1ArDHDg91UahHFsMxbKWycYsjze4muTsP64vT43v7hSf/A0FgdjQPQWAmco68nB+T+SFSqNUQgcIbN1bn8Z3RwvL22MAvcu8TACFgrpMVZ4aUYcn77BMDkxGgemAGOHIBXxRjBWZMKoCPA2h6qEUSRR2MF6GxUUMUaIUgBCNTnAcm3H2G5YQfgvccYIXAtDH7FoKq/AaqKlbrBj2trFVXfBPAea4SOIIsBeN9kkCwxsNkAqRWy7+B7Z00G3xVc2wZeMSI4S7sVYkSk5Z/4PyBWROqvox3A28PN2cjUwinQC9QyckKALxj4kv2auK0xAAAAAElFTkSuQmCC'



starting_path = sg.popup_get_folder('Folder to display')

if not starting_path:
    sys.exit(0)

treedata = sg.TreeData()


def add_files_in_folder(parent, dirname):
    """
    TreeDataに値を格納するための関数
    """
    files = os.listdir(dirname)
    for f in files:
        fullname = os.path.join(dirname, f) # ファイル名を取得
        if os.path.isdir(fullname): # フォルダーの場合は、フォルダーを追加して再帰します
            treedata.Insert(parent, fullname, f, values=[], icon=folder_icon)
        else:

            treedata.Insert(parent, fullname, f, values=[
                            os.stat(fullname).st_size], icon=file_icon)


add_files_in_folder('', starting_path)


layout = [[sg.Text('File and folder browser Test')],
          [sg.Tree(data=treedata,
                   headings=['Size', ],
                   auto_size_columns=True,
                   num_rows=20,
                   col0_width=30,
                   key='-TREE-',
                   show_expanded=False,
                   enable_events=True),
           ],
          [sg.Button('Ok'), sg.Button('Cancel')]]

window = sg.Window('Tree Element Test', layout)


while True:     # Event Loop
    event, values = window.read()
    if event in (None, 'Cancel'):
        break
    print(event, values)
window.close()