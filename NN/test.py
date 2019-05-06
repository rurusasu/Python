import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.layers import *
from collections import OrderedDict

#dict = {{"x1":1, "x2":2}, {"y1":1, "y2":2}} Error
dict = {"input":2, "Dense":50, "Dense":20, "Dense":1} #画面表示結果 {'input': 2, 'Dense': 1}
dict = {"input":2, "Dense1":50, "Dense2":20, "Dense3":1}  #画面表示結果 {'input': 2, 'Dense1': 50, 'Dense2': 20, 'Dense3': 1}
print(dict)

dict = {"x1":1, "y1":2}
for i in dict.values():
    print(i)

List = [{"x1":1}, {"y1":2}] #リストの中にはdictオブジェクトを入れることができる。
for i in List:
    #print(i.values())
    print(i)

file = {}
for i in List:              #リスト内のdictオブジェクトは次の方法で一つのdictオブジェクトに統合できる。
    file.update(i)
    print(file)

class test:
    def __init__(self, x):
        self.x = x
        #return self.x       #__init__ではreturnの際に値を返すべきではない
    def add(self, y):
        z = self.x + y

        return z

x = test(2)
y = x.add(3)
print(x, y)