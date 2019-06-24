import numpy as np


# 求めたいグラフ
def func(x, y):
    a=1
    b=2
    c=1
    d=1
    e=1
    f=1

    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

#data = np.arange(1, 4, 1)
data_x = np.arange(-100, 101, 1)
data_y = np.arange(-100, 101, 1) 
data_z = []

counter = 0

for i in data_x:
    for j in data_y:
        z = func(i, j)
        num = np.array([i, j, z])
        
        if (counter == 0):
            data = num
        else:
            data = np.vstack((data, num))
        #data = np.vstack((data, num))
        #data.append(num)
        counter += 1

# メイン関数
# 重さの定義
w1 = (np.ones((1, 50)) - np.random.rand(1, 50))

# バイアスの定義
b1 = (np.ones(50) - np.random.rand(50))


for i in range(100000):
    Batch_mask = np.random.choice(data.shape[0], 100, replace = False)
    tid = data[Batch_mask, 0:2] # tid:traning input data
    ttd = data[Batch_mask, 2]   # ttd:traning test data
    
    for j in range(Batch_mask.size):
        # Layer1
        input_data = tid[j, :]
        input1 = np.dot(tid, w1) + b1

