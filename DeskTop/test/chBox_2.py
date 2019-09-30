# cording: utf-8

import tkinter as tk


# ウインドウを作成する
win = tk.Tk()
win.title('CheckTitle')
win.geometry('400x200')


chk_txt = ['JogMode', 'CoordinateMode'] # チェックボタンのラベルをリスト化する
chk_bln = {} # チェックボックスのON/OFF状態

# チェックボタンを動的に作成して配置
for i in range(len(chk_txt)):
    chk_bln[i] = tk.BooleanVar() # チェックボックスの各項目の初期値
    chk = tk.Checkbutton(win, variable=chk_bln[i], text=chk_txt[i])
    chk.place(x = 50, y = 30 + (i * 24))

# ボタンクリックイベント（チェック有無をセット）
def btn_click(bln):
    for i in range(len(chk_bln)):
        chk_bln[i].set(bln)


btn = tk.Button(win, text='ONにする', command=lambda:btn_click(True))
btn.place(x=80, y=170)
btn = tk.Button(win, text='OFFにする', command=lambda:btn_click(False))
btn.place(x=150, y=170)

tk.mainloop()