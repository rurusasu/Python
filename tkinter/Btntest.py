import tkinter as tk

win = tk.Tk()

# ラベルウィジットの定義
lbl = tk.Label(win)
lbl['text'] = '1'
lbl.pack(fill='x', padx=20, side='top')

# ラベル
self.lrnDlbl = tk.Label(self)
self.lrnDlbl['text'] = 'learn'
self.lrnDlbl.pack(fill='x', padx=20, side='top')
# tBox
self.lrnDtxt = tk.Entry(width=13)
self.lrnDtxt.pack(fill='x', padx=20, side='top')
self.lrnDtxt.insert(tk.END, 'learn.csv')






win.mainloop()
