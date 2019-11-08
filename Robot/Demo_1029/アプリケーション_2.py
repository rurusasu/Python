# cording: utf-8

import threading
import DobotDllType as dType
from ctypes import * # cdllを呼ぶために必要

# ウインドウ作成に必要
import tkinter as tk
from tkinter import messagebox as mbox
from tkinter import Checkbutton as cbutton
from common.csvIO import csvIO
from nn import nn
from common.DobotFunction import*

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

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_ConnectBtn(self):
        self.CntBtn = tk.Button(self)
        self.CntBtn['text'] = 'Connect'
        self.CntBtn['command'] = connect_click
    
    def create_ActPlace(self):
        self.ActBtn = tk.Button(self)
        self.ActBtn['text'] = 'DataGet'
        self.ActBtn['command'] = DetaGet_click

        # ラベル
        self.Savlbl = tk.Label(self)
        self.Savlbl['text'] = 'SaveFile'
        self.Savlbl.pack(fill='x', padx=20, side='top')
        # tBox
        self.Savtxt = tk.Entry(width=13)
        self.Savtxt.pack(fill='x', padx=20, side='top')
        self.Savtxt.insert(tk.END, 'data.csv')  # テキストボックスに文字をセット

    def create_ValPlace(self):
        self.ValBtn = tk.Button(self)
        self.ValBtn['text'] = 'Validation'
        self.ValBtn['command'] = ValDataGet_click

        # ラベル
        self.Vallbl = tk.Label(self)
        self.Vallbl['text'] = 'Validation'
        self.Vallbl.pack(fill='x', padx=20, side='top')
        # tBox
        self.Valtxt = tk.Entry(width=13)
        self.Valtxt.pack(fill='x', padx=20, side='top')
        self.Valtxt.insert(tk.END, 'val.csv')

    def create_DConvPlace(self):
        self.CnvBtn = tk.Button(self)
        self.CnvBtn['text'] = 'DataConv'
        self.CnvBtn['command'] = DataConv_click

        # タイトルラベル
        self.Cnvlbl = tk.Label(self)
        self.Cnvlbl['text'] = 'DataConv'
        self.Cnvlbl.pack(fill='x', padx=20, side='top')

        #----------------------------
        # 元データ
        #----------------------------
        # ラベル
        self.orgDlbl = tk.Label(self)
        self.orgDlbl['text'] = 'Original Data'
        self.orgDlbl.pack(fill='x', padx=20, side='top')
        # tBox
        self.orgDtxt = tk.Entry(width=13)
        self.orgDtxt.pack(fill='x', padx=20, side='top')
        self.orgDtxt.insert(tk.END, 'data.csv')
        
        #----------------------------
        # 学習用データ
        #----------------------------
        # ラベル
        self.lrnDlbl = tk.Label(self)
        self.lrnDlbl['text'] = 'learn'
        self.lrnDlbl.pack(fill='x', padx=20, side='top')
        # tBox
        self.lrnDtxt = tk.Entry(width=13)
        self.lrnDtxt.pack(fill='x', padx=20, side='top')
        self.lrnDtxt.insert(tk.END, 'learn.csv')

        #----------------------------
        # テスト用データ
        #----------------------------
        # ラベル
        self.tstDlbl = tk.Label(self)
        self.tstDlbl['text'] = 'test'
        self.tstDlbl.pack(fill='x', padx=20, side='top')
        # tBox
        self.lrnDtxt = tk.Entry(width=13)
        self.lrnDtxt.pack(fill='x', padx=20, side='top')
        self.lrnDtxt.insert(tk.END, 'test.csv')

        #----------------------------
        # 小数点以下の桁数
        #----------------------------
        # ラベル
        self.RudUplbl = tk.Label(self)
        self.RudUplbl['text'] = '小数点以下の桁数'
        self.RudUplbl.pack(fill='x', padx=20, side='top')
        # tBox
        self.RudUptxt = tk.Entry(width=3)
        self.RudUptxt.pack(fill='x', padx=20, side='top')
        self.RudUptxt.insert(tk.END, '2')

    def create_NeuralNetPlace(self):
        self.NNbtn = tk.Button(self)
        self.NNbtn['text'] = 'NN'
        self.NNbtn.pack(fill='x', padx=20, side='top')


def __filePath__(file_name):
    path = './data/' + str(file_name)
    return path


def DataInput():
    xPath = __filePath__(txt_6.get())
    tPath = __filePath__(txt_7.get())

    io = csvIO()
    x = io.open_twoD_array(xPath)
    t = io.open_twoD_array(tPath)

    x = io.twoD_FroatToStr(x, digit=0.01)
    t = io.twoD_FroatToStr(t, digit=0.01)
    x = io.twoD_Numpy(x)
    t = io.twoD_Numpy(t)
    print('読み込みに成功')

    return x, t





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
    initDobot(api)
   

def DetaGet_click():
    x_roop = 100
    y_roop = 100
    z_roop = 2

    file_name=__filePath__(txt.get())
    initPOS = dType.GetPose(api)
    #-----------------------------
    # 以下Z軸方向の動作
    #-----------------------------
    for i in range(0, z_roop):
        print('第' + str(i + 1) + 'ステップ目')
        Operation(api, file_name, 'z', -i, initPOS)

        #-------------------------
        # 以下Y軸方向の動作
        #-------------------------
        for j in range(0, y_roop):
            Operation(api, file_name, 'y')

            #-------------------------
            # 以下X軸方向の動作
            #-------------------------
            if j % 2 == 0:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Operation(api, file_name, 'x')
            else:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Operation(api, file_name, 'x', -1)

    print('データ取得が終了しました。')


def ValDataGet_click():
    x_roop = 100
    y_roop = 100
    z_roop = 2

    file_name = __filePath__(txt_10.get())
    initPOS = dType.GetPose(api)
    #-----------------------------
    # 以下Z軸方向の動作
    #-----------------------------
    for i in range(0, z_roop):
        print('第' + str(i + 1) + 'ステップ目')
        Operation(api, file_name, 'z', -0.3*i, initPOS)

        #-------------------------
        # 以下Y軸方向の動作
        #-------------------------
        for j in range(0, y_roop):
            Operation(api, file_name, 'x', 0.3)

            #-------------------------
            # 以下X軸方向の動作
            #-------------------------
            if j % 2 == 0:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Operation(api, file_name, 'y', 0.3)
            else:
                for k in range(0, x_roop + 1):
                    #Async Motion Params Setting
                    Operation(api, file_name, 'y', -0.3)

    print('testデータ取得が終了しました。')


def DataConv_click():
    inputPath = __filePath__(txtConv_1.get())
    learnPath = __filePath__(txtConv_2.get())
    testPath = __filePath__(txtConv_3.get())
    digit = str(10**-int(txtConv_4.get()))

    io = csvIO()
    v = io.open_twoD_array(inputPath)
    v = io.twoD_FroatToStr(v, digit=digit)
    l_array = io.Get_AnytwoD_array(v, col_range_end=3)
    t_array = io.Get_AnytwoD_array(v, col_range_first=4, col_range_end=7)

    io.csv_write(learnPath, l_array)
    io.csv_write(testPath, t_array)


def NN_click():
    batch_size = int(txt_8.get())
    epochs = int(txt_9.get())
    x, t = DataInput()

    nn(x, t, batch_size, epochs, rdo_click())


# ウインドウの作成
win = tk.Tk()
win.title('Dobot')      # タイトル
win.geometry('500x400')  # サイズを指定


#--------------------------------
# connect
#--------------------------------
# connect用
# ボタン
connectBtn = tk.Button(win, text='connect',
                       command=connect_click, width=10)  # ボタンを作成
connectBtn.place(x=20, y=10)


#--------------------------------
# Save
#--------------------------------
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
# validationデータの取得
#--------------------------------
Dcon_x = 33
Dcon_y = 120
# Validation用ラベル
lbl_10 = tk.Label(text='Validation')
lbl_10.place(x=Dcon_x, y=Dcon_y)
# tBox
txt_10 = tk.Entry(width=13)
txt_10.place(x=Dcon_x-13, y=Dcon_y+20)
txt_10.insert(tk.END, 'val.csv')
# ボタン
datamakeBtn = tk.Button(win, text='Validation',
                        command=ValDataGet_click, width=10)
datamakeBtn.place(x=Dcon_x-13, y=Dcon_y+40)


#--------------------------------
# DataConv
#--------------------------------
Dcon_x = 33
Dcon_y = 190
# データ変換部分用ラベル
lblConv_1 = tk.Label(text='DataConv')
lblConv_1.place(x=Dcon_x, y=Dcon_y)

# 元データ
# ラベル
lblConv_2 = tk.Label(text='LowData')
lblConv_2.place(x=Dcon_x-13, y=Dcon_y+20)
# tBox
txtConv_1 = tk.Entry(width=13)
txtConv_1.place(x=Dcon_x-13, y=Dcon_y+40)
txtConv_1.insert(tk.END, 'data.csv')

# 学習用データ
Dcon_y = Dcon_y + 40
# ラベル
lblConv_3 = tk.Label(text='learn')
lblConv_3.place(x=Dcon_x-13, y=Dcon_y+20)
# tBox
txtConv_2 = tk.Entry(width=13)
txtConv_2.place(x=Dcon_x-13, y=Dcon_y+40)
txtConv_2.insert(tk.END, 'learn.csv')

# テスト用データ
Dcon_y = Dcon_y + 40
# ラベル
lblConv_4 = tk.Label(text='test')
lblConv_4.place(x=Dcon_x-13, y=Dcon_y+20)
# tBox
txtConv_3 = tk.Entry(width=13)
txtConv_3.place(x=Dcon_x-13, y=Dcon_y+40)
txtConv_3.insert(tk.END, 'test.csv')

# 小数点以下の桁数
# ラベル
lblConv_5 = tk.Label(text='小数点以下の桁数')
lblConv_5.place(x=Dcon_x+50, y=Dcon_y+20)
# tBox
txtConv_4 = tk.Entry(width=3)
txtConv_4.place(x=Dcon_x+80, y=Dcon_y+40)
txtConv_4.insert(tk.END, '2')

# DataMake
Dcon_y = Dcon_y + 40
# ボタン
datamakeBtn = tk.Button(win, text='DataMake', command=DataConv_click, width=10)
datamakeBtn.place(x=Dcon_x-13, y=Dcon_y+20)


#--------------------------------
# NuralNet
#--------------------------------
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
nn_y = nn_y + 40
# ラベル
lbl_7 = tk.Label(text='test')
lbl_7.place(x=nn_x, y=nn_y+20)
# tBox
txt_7 = tk.Entry(width=13)
txt_7.place(x=nn_x, y=nn_y+40)
txt_7.insert(tk.END, 'test.csv')

# バッチ
nn_y += 40
# ラベル
lbl_8 = tk.Label(text='batch')
lbl_8.place(x=nn_x, y=nn_y+20)
# tBox
txt_8 = tk.Entry(width=5)
txt_8.place(x=nn_x, y=nn_y+40)
txt_8.insert(tk.END, '128')

# エポック
# ラベル
lbl_9 = tk.Label(text='epoch')
lbl_9.place(x=nn_x+40, y=nn_y+20)
# tBox
txt_9 = tk.Entry(width=5)
txt_9.place(x=nn_x+40, y=nn_y+40)
txt_9.insert(tk.END, '100')

# Feature Scaling
nn_y += 70
# ラジオボタン
rdo_var = tk.IntVar() # チェック有無変数
rdo_var.set(2) # チェックの初期位置
rdo_txt=['標準化', '正規化', '両方']
for i in range(len(rdo_txt)):
    #chk_bln[i] = tk.BooleanVar()
    rdo = tk.Radiobutton(win, variable=rdo_var, value=i, text=rdo_txt[i])
    rdo.place(x=nn_x, y=nn_y + (i*20))

def rdo_click():
    num = rdo_var.get()
    return num


# NuralNetの実行
nn_y = nn_y + 40
nnBtn = tk.Button(win, text='NN', command=NN_click, width=10)
nnBtn.place(x=nn_x, y=nn_y+20)


#--------------------------------
# 移動範囲
#--------------------------------
act_x = 300
act_y = 140
# x方向の移動範囲
# ラベル
lbl_x = tk.Label(text='x=')
lbl_x.place(x=act_x, y=act_y)
lbl_moji1 = tk.Label(text='から')
lbl_moji1.place(x=act_x+50, y=act_y)
# tBox
txt_x1 = tk.Entry(width=4)
txt_x1.place(x=act_x+20, y=act_y)
txt_x2 = tk.Entry(width=4)
txt_x2.place(x=act_x+75, y=act_y)

# y方向の移動範囲
act_y = act_y+25
# ラベル
lbl_y = tk.Label(text='y=')
lbl_y.place(x=act_x, y=act_y)
lbl_moji2 = tk.Label(text='から')
lbl_moji2.place(x=act_x+50, y=act_y)
# tBox
txt_y1 = tk.Entry(width=4)
txt_y1.place(x=act_x+20, y=act_y)
txt_y2 = tk.Entry(width=4)
txt_y2.place(x=act_x+75, y=act_y)

# z方向の移動範囲
act_y = act_y+25
# ラベル
lbl_z = tk.Label(text='z=')
lbl_z.place(x=act_x, y=act_y)
lbl_moji3 = tk.Label(text='から')
lbl_moji3.place(x=act_x+50, y=act_y)
# tBox
txt_z1 = tk.Entry(width=4)
txt_z1.place(x=act_x+20, y=act_y)
txt_z2 = tk.Entry(width=4)
txt_z2.place(x=act_x+75, y=act_y)



win.mainloop()          # ウインドウを動かす
