# cording: utf-8

import tkinter as tk
from tkinter import  messagebox as messagebox

# ウインドウを作成
win = tk.Tk()
win.title('Software Title')
win.geometry('500x200')

#------------------
# グローバル変数
#------------------
hLabel   = [] # ラベルのハンドルを格納する
hCheck   = [] # チェックボックスのハンドルを格納する
CheckVal = [] # チェックボックスにチェックが入っているかを格納する


#-------------------------------
# チェックボックスの状況を取得する
#-------------------------------
def check(event):
    for n in range(len(CheckVal)):
        if CheckVal[n].get() == True:
            label = tk.Label(text='チェックされています')
            label.place(x=100, y=20*n + 50)
        else:
            label = tk.Label(text='チェックされていません')
            label.place(x=100, y=20*n + 50)

        # ラベルのハンドルを追加
        hLabel.append(label)


#---------------------------
# チェックボックスを動的に作成
#---------------------------
def make(event):
    # 作成するチェックボックスの個数(Entryの値)を取得
    num = Entry1.get()

    # 既出のチェックボックスやラベルを削除
    for n in range(len(hCheck)):
        hCheck[n].destroy()
        hLabel[n].destroy()

    # 配列を空にする
    del CheckVal[:]
    del hCheck[:]
    del hLabel[:]

    # Entry1に入力された値分ループ
    for n in range(int(num)):
        #BooleanVarの作成
        bln = tk.BooleanVar()

        # チェックボックスの値を決定
        bln.set(False)

        # チェックボックスの作成
        b = tk.Checkbutton(text='項目' + str(n+1), variable=bln)
        b.place(x=5, y=20*n + 50)

        # チェックボックスの値をリストに追加
        CheckVal.append(bln)

        # チェックボックスのハンドルをリストに追加
        hCheck.append(b)


#button1 = tk.Button(win, width=20, command=make, text='CheckButtonの作成')
button1 = tk.Button(win, width=20, text='CheckButtonの作成')
button1.bind('<Button-1>', make)
button1.place(x=90, y=5)

#button2 = tk.Button(win, width=15, command=check, text='チェックの心得')
button2 = tk.Button(win, width=15, text='チェックの取得')
button2.bind('<Button-1>', check)
button2.place(x=265, y=5)

Entry1 = tk.Entry(win, width=10)
Entry1.place(x=5, y=5)

win.mainloop()
