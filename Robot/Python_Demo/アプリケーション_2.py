# cording: utf-8

import threading
import DobotDllType as dType
from ctypes import * # cdllを呼ぶために必要

# ウインドウ作成に必要
import tkinter as tk
from tkinter import messagebox as mbox
from tkinter import Checkbutton as cbutton
from common.csvIO import csvIO
from DobotFunction import*
import csv

# Load Dll
api = cdll.LoadLibrary("DobotDll.dll")


CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

#-------------------
# グローバル変数
#-------------------
hLabel   = [] # ラベルのハンドルを格納する
hCheck   = [] # チェックボックスのハンドルを格納する
CheckVal = [] # チェックボックスにチェックが入っているかを格納する


def __filePath__(file_name):
    path = './data/' + str(file_name)
    return path


#------------------------
# ボタンが押されたときの処理
#------------------------
# connectButtonが押されたときの処理
def connect_click():
    # Dobot Connect
    state = dType.ConnectDobot(api, "", 115200)[0] # ConectDobot(const char* pointName, int baudrate)
    if (state != dType.DobotConnect.DobotConnect_NoError):
        mbox.showinfo('Dobot Connect', 'Dobotに接続できませんでした。')
    
    #Clean Command Queued
    dType.SetQueuedCmdClear(api)

    #Async Motion Params Setting
    dType.SetHOMEParams(api, 150, -200, 100, 0, isQueued=1)

    #Async Home
    dType.SetHOMECmd(api, temp=0, isQueued=1)

    #キューに入っているコマンドを実行
    dType.SetQueuedCmdStartExec(api)

    #キューに入っているコマンドを停止
    dType.SetQueuedCmdStopExec(api)

    # Dobotの初期設定
    initDobot()
   

def DetaGet_click():
    """
    x_roop1 = 4
    x_roop2 = 20
    y_roop = 100
    z_roop = 2
    counter_x = 150
    counter_x_init = 150
    counter_y = -201
    counter_y_init = -201
    counter_z = 101
    """
    x_roop1 = 4
    x_roop2 = 2
    y_roop = 2
    z_roop = 2
    counter_x = 150
    counter_x_init = 150
    counter_y = -201
    counter_y_init = -201
    counter_z = 101

    file_name=__filePath__(txt.get())
    #-----------------------------
    # 以下Z軸方向の動作
    #-----------------------------
    for i in range(0, z_roop):
        print('第' + str(i + 1) + 'ステップ目')
        Operation(api, 'z', -1)


        #-------------------------
        # 以下Y軸方向の動作
        #-------------------------
        for j in range(0, y_roop):
            Operation(api, 'y')

            #-------------------------
            # 以下X軸方向の動作
            #-------------------------
            if j % 2 == 0:
                for k in range(0, x_roop1 + 1):

                    #Async Motion Params Setting
                    dType.SetPTPJointParams(
                        api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1)
                    dType.SetPTPCommonParams(api, 100, 100, isQueued=1)
                    Operation(api, 'x')
                    csv_write(file_name, dType.GetPose(api))
            else:
                for k in range(0, x_roop1 + 1):

                    #Async Motion Params Setting
                    dType.SetPTPJointParams(
                        api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1)
                    dType.SetPTPCommonParams(api, 100, 100, isQueued=1)
                    Operation(api, 'x', -1)
                    csv_write(file_name, dType.GetPose(api))
        counter_x = counter_x_init
        counter_y = counter_y_init

    pose = dType.GetPose(api)


def DetaMake_click():
    inputPath = __filePath__(txt.get())
    learnPath = __filePath__(txt_2.get())
    testPath = __filePath__(txt_3.get())
    digit = str(10^int(txt_4.get()))

    io = csvIO()
    v = io.open_twoD_array(inputPath)
    v = io.twoD_FroatToStr(v, digit=digit)
    l_array = io.Get_AnytwoD_array(v, col_range_end=3)
    t_array = io.Get_AnytwoD_array(v, col_range_first=4, col_range_end=7)

    io.csv_write(learnPath, l_array)
    io.csv_write(testPath, t_array)


#def NN_click()


# ウインドウの作成
win = tk.Tk()
win.title('Dobot')      # タイトル
win.geometry('500x400')  # サイズを指定


#------------
# Save
#------------
Sfile_x = 33
Sfile_y = 50
# テキストボックス用
# ラベル
lbl = tk.Label(text='SaveFile')
lbl.place(x=Sfile_x, y=Sfile_y)
# tBox
txt = tk.Entry(width=13)
txt.place(x=Sfile_x-13, y=Sfile_y+20)
txt.insert(tk.END, 'data.csv')  # テキストボックスに文字をセット
# ボタン
datagetBtn = tk.Button(win, text='DataGet', command=DetaGet_click, width=10)
datagetBtn.place(x=Sfile_x-13, y=Sfile_y+40)

#--------------------------------
# DataConv
#--------------------------------
Dcon_x = 33
Dcon_y = 140
# データ変換部分用ラベル
lbl_2 = tk.Label(text='DataConv')
lbl_2.place(x=Dcon_x, y=Dcon_y)

# 学習用データ
# ラベル
lbl_3 = tk.Label(text='learn')
lbl_3.place(x=Dcon_x-13, y=Dcon_y+20)
# tBox
txt_2 = tk.Entry(width=13)
txt_2.place(x=Dcon_x-13, y=Dcon_y+40)
txt_2.insert(tk.END, 'learn.csv')

# テスト用データ
# ラベル
lbl_4 = tk.Label(text='test')
lbl_4.place(x=Dcon_x-13, y=Dcon_y+60)
# tBox
txt_3 = tk.Entry(width=13)
txt_3.place(x=Dcon_x-13, y=Dcon_y+80)
txt_3.insert(tk.END, 'test.csv')

# 小数点以下の桁数
# ラベル
lbl_5 = tk.Label(text='小数点以下の桁数')
lbl_5.place(x=Dcon_x+50, y=Dcon_y+60)
# tBox
txt_4 = tk.Entry(width=3)
txt_4.place(x=Dcon_x+80, y=Dcon_y+80)
txt_4.insert(tk.END, '2')

# DataMake
# ボタン
datamakeBtn = tk.Button(win, text='DataMake', command=DetaMake_click, width=10)
datamakeBtn.place(x=Dcon_x-13, y=Dcon_y+100)


#-----------------------
# NuralNet
#-----------------------
nn_x = 200
nn_y = 0
txt_5 = tk.Label(text='NuralNet')
txt_5.place(x=nn_x, y=nn_y)
# 学習用データ
# ラベル
lbl_6 = tk.Label(text='learn')
lbl_6.place(x=nn_x, y=nn_y+20)
# tBox
txt_6 = tk.Entry(width=13)
txt_6.place(x=nn_x, y=nn_y+40)
txt_6.insert(tk.END, 'learn.csv')

# テスト用データ
# ラベル
lbl_7 = tk.Label(text='test')
lbl_7.place(x=nn_x, y=nn_y+60)
# tBox
txt_7 = tk.Entry(width=13)
txt_7.place(x=nn_x, y=nn_y+80)
txt_7.insert(tk.END, 'test.csv')

#------------
# ボタン
#------------
connectBtn = tk.Button(win, text='connect', command=connect_click, width=10) # ボタンを作成
connectBtn.place(x=20, y=10)


win.mainloop()          # ウインドウを動かす
