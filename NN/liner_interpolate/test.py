import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from MyFunction import load_data
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

def test5():
    #output  = []
    output  = {}
    history = {}
    history['loss'] = None
    history['acc']  = None
    #output.append(history)
    output['history'] = history
    
    return output

#a = test5()
#print(a.history) #historyがkeyとして登録されているので、表示したい場合はa['history']にする必要がある。

class output: pass

class test6C:
    def __init__(self):
        self.Output = output()
        self.Output.history = {}

        self.Output.history['loss'] = []
        self.Output.history['acc']  = []

    def test6(self):
        self.Output.history['loss'].append([0, 0])
        self.Output.history['acc'].append([1, 1])

        return self.Output

a = test6C()
b = a.test6()
c = b.history['loss']
print(b)
print(c)


#(x_train, t_train),(x_test, t_test) = load_data("C:/Data/H31_Miki/lab/NN/MNIST")

#fig = plt.figure(figsize = (9, 9))
#fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 0.5, hspace = 0.05, wspace = 0.05)

#for i in range(81):
    #ax = fig.add_subplot(9, 9, i+1, xticks = [], yticks = [])
    #ax.imshow(x_train[i].reshape((28, 28)), cmap = 'gray')

#plt.show()


#Xavierの一様分布
node_num = 100 #前層のノードの数
w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
print(w)