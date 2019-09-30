# cording: utf-8

import tkinter as tk
from tkinter import messagebox as mbox

# ウインドウを作成
win = tk.Tk()
win.geometry("500x250") # サイズを指定

# 部品を作成
# ラベルを作成
label = tk.Label(win, text='名前は？')
label.pack()

# テキストボックスを作成
text = tk.Entry(win)
text.pack()
text.insert(tk.END, 'クジラ') # 初期値を指定

# OKボタンを押したとき
def ok_click():
    # テキストボックスの内容を得る
    s = text.get()
    # ダイアログを表示
    mbox.showinfo('挨拶', s + 'さん、こんにちは！')

# ボタンを作成
okButton = tk.Button(win, text='OK', command=ok_click)
okButton.pack()

# ウインドウを動かす
win.mainloop()