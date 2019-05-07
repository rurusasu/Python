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


A = np.array([[0.59185423],[0.11811593],[1.1497815]])
B1 = np.array([[0.21611296],[0.07906977],[0.01568106]])
C1 = 0.5*np.sum((A - B1)**2, axis = 0)
C2 = 0.5*np.sum((A - B1)**2, axis = 1)
C3 = 0.5*(A - B1)**2

print('C1 = ', C1, C1.dtype, C1.shape)
print('C2 = ', C2, C2.dtype, C2.shape)
print('C3 = ', C3, C3.dtype, C3.shape)

B2 = np.array([0.21611296, 0.07906977, 0.01568106])

D1 = 0.5*np.sum((A - B2)**2, axis = 0)
D2 = 0.5*np.sum((A - B2)**2, axis = 1)
D3 = 0.5*(A - B2)**2

print('D1 = ', D1, D1.shape)
print('D2 = ', D2, D2.shape)
print('D3 = ', D3, D3.shape)

B3 = B2.reshape(3, -1)

E1 = 0.5*np.sum((A - B3)**2, axis = 0)
E2 = 0.5*np.sum((A - B3)**2, axis = 1) #成功
E3 = 0.5*(A - B3)**2                   #成功

print('E1 = ', E1, E1.shape)
print('E2 = ', E2, E2.shape)
print('E3 = ', E3, E3.shape)


print('A = ', A.shape)
print('B1 = ', B1.shape)
print('B2 = ', B2.shape)
print('B3 = ', B3.shape)


class test:
    count = 0
    def __init__(self):
        self.sequential = OrderedDict()

    def add(self, layer_name):
        key = 'layer' + str(test.count)
        #dic = dict(zip(key, layer_name))
        dic = dict(key, layer_name)
        self.sequential.update(dic)

        test.count += 1

class test2:
    def __init__(self, units, activation):
        self.dense = OrderedDict()
        self.params = {}

        self.params['Weight'] = None
        self.params['Bias']   = None
        self.activation = activation

#class test3: OrderedDict()   #add関数が存在しないためエラー

class test4:
    def __init__(self):
        pass

    #def add(self, layer_name): OrderedDict()
    def add(self, layer_name): pass

module = test4()
module.add(test2(10, 'liner'))

print (module.__dict__)