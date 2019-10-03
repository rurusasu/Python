# cording: utf-8

import threading
import DobotDllType as dType
from ctypes import * # cdllを呼ぶために必要

# ウインドウ作成に必要
import tkinter as tk
from tkinter import messagebox as mbox
from tkinter import Checkbutton as cbutton
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


#-----------------
# Dobotの初期化
#-----------------
def initDobot():
    # Clean Command Queued
    dType.SetQueuedCmdClear(api)

    # デバイスのシリアルナンバーを取得する
    dSN = dType.GetDeviceSN(api)
    print(dSN)

    # デバイス名を取得する
    dName = dType.GetDeviceName(api)
    print(dName)

    # デバイスのバージョンを取得する
    majorV, minorV, revision = dType.GetDeviceVersion(api)
    print(majorV, minorV, revision)

    # JOGパラメータの設定
    dType.SetJOGJointParams(api, 200, 200, 200, 200,
                            200, 200, 200, 200, isQueued=1)      # 関節座標系での各モータの速度および加速度の設定
    dType.SetJOGCoordinateParams(api, 200, 200, 200, 200,
                                 200, 200, 200, 200, isQueued=1)  # デカルト座標系での各方向への速度および加速度の設定
    # JOG動作の速度、加速度の比率を設定
    dType.SetJOGCommonParams(api, 100, 100, isQueued=1)

    # PTPパラメータの設定
    dType.SetPTPJointParams(api, 200, 200, 200, 200,
                            200, 200, 200, 200, isQueued=1)           # 関節座標系の各モータの速度および加速度を設定
    # デカルト座標系での各方向への速度および加速度の設定
    dType.SetPTPCoordinateParams(api, 200, 200, 200, 200, isQueued=1)
    # PTP動作の速度、加速度の比率を設定
    dType.SetPTPCommonParams(api, 100, 100, isQueued=1)


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
                    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1)
                    dType.SetPTPCommonParams(api, 100, 100, isQueued=1)      
                    Operation(api, 'x')
                    csv_write('data.csv', dType.GetPose(api))
            else:
                for k in range(0, x_roop1 + 1):

                    #Async Motion Params Setting
                    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1)
                    dType.SetPTPCommonParams(api, 100, 100, isQueued=1)      
                    Operation(api, 'x', -1)
                    csv_write('data.csv', dType.GetPose(api))
        counter_x = counter_x_init
        counter_y = counter_y_init


    pose = dType.GetPose(api)
    
    

    # Dobotの初期設定
    initDobot()
   

# ウインドウの作成
win = tk.Tk()
win.title('Dobot')      # タイトル
win.geometry('500x250')  # サイズを指定


#------------
# ボタン
#------------
connectBtn = tk.Button(win, text='connect', command=connect_click) # ボタンを作成
connectBtn.pack()

datagetBtn = tk.Button(win, text='DataGet', )

win.mainloop()          # ウインドウを動かす
