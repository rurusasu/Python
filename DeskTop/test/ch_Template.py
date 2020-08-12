# cording: utf-8

import tkinter as tk

#----------------------------------
# チェックボックスのテンプレート
#----------------------------------


#-------------------
# グローバル変数
#-------------------
hLabel   = [] # ラベルのハンドルを格納する
hCheck   = [] # チェックボックスのハンドルを格納する
CheckVal = [] # チェックボックスにチェックが入っているかを格納する


#-------------------------
# チェックボックスの項目
#-------------------------
chk_txt = ['JogMode', 'CoordinateMode']


#-----------------------------------
# チェックボックスを動的に作成
#-----------------------------------
def chmake(event):
    # 既出のチェックボックスやラベルを削除
    for n in range(len(hCheck)):
        hCheck[n].destroy()
        hLabel[n].destroy()

    # 配列を空にする
    del CheckVal[:]
    del hCheck[:]
    del hLabel[:]

    # Entry1に入力された値分ループ
    ch_MakeAndSet(chk_txt, CheckVal, hCheck)


#---------------------------------------
# チェックボックスの作成と配置
#---------------------------------------
def ch_MakeAndSet(chk_txt, CheckVal, hCheck):
    for i in range(len(chk_txt)):
        # チェックボックスの項目の初期値
        bln = tk.BooleanVar()

        # チェックボックスの値を決定
        bln.set(False)

        # チェックボックスの作成
        b = tk.Checkbutton(win, variable = bln, text = chk_txt[i])
        b.place(x=5, y=20*i + 50)

        # チェックボックスの値をリストに追加
        CheckVal.append(bln)
        
        # チェックボックスのハンドルをリストに追加
        hCheck.append(b)


#-----------------------------------
# チェックボックスの状況を取得する
#-----------------------------------
def chget(event):
    for n in range(len(CheckVal)):
        if CheckVal[n].get() == True: 
            # 項目にチェックが付いているときの処理
            label = tk.Label(text='チェックされています')
            label.place(x=100, y=20*n + 50)

        else:                         
            # 項目にチェックが付いていないときの処理
            label = tk.Label(text='チェックされていません')
            label.place(x=100, y=20*n + 50)

        # ラベルのハンドルを追加
        hLabel.append(label)


# ウインドウの作成
win = tk.Tk()
win.title('chTemplate')  # タイトル
win.geometry('500x250')  # サイズを指定

button1 = tk.Button(win, width=20, text='CheckButtonの作成')
button1.bind('<Button-1>', chmake)
button1.place(x=90, y=5)

button2 = tk.Button(win, width=15, text='チェックの取得')
button2.bind('<Button-1>', chget)
button2.place(x = 265, y=5)

win.mainloop()