import numpy as np

x = np.random.normal(10, 5, (128, 2))
t = np.random.normal(10, 5, (128, 1))
W1 = np.ones((2, 50))
W2 = np.ones((50, 1))
B1 = np.zeros(50)
B2 = np.zeros(1)

#出力層
#2乗和誤差レイヤ
class mean_squared_error:
    def __init__(self):
        self.loss = None #損失
        self.y    = None #linerの出力
        self.t    = None #教師データ

    def forward(self, x, t):
        self.y = x
        self.t = t
        if t.shape != x.shape:
            self.y = self.y.reshape(self.y.size, 1)
            self.t = self.t.reshape(self.t.size, 1)
        self.loss = 0.5 * np.sum((self.y - self.t)**2)

        return self.loss

    def backward(self, dout = 1):
        #if dout != 1:
        if dout == 1:
            batch_size = self.t.shape[0]
            if self.y.size == self.t.size:
                dout = (self.y - self.t) / batch_size

        return dout


#恒等関数レイヤ
class liner:
    def __init__(self):
        self.x = None 

    def forward(self, x):
        self.x = x
        return self.x

    def backward(self, dout):
        delta = 1 * dout

        return delta


#ReLUレイヤ
class relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0) #xが0以下ならFalseを返す(0より大きければTrue)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


#Affine
class affine:
    def __init__(self, W, b):
        self.W = W
        self.B = b

        self.x = None
        self.original_x_shape = None
        #重み・バイアスパラメータの微分
        self.dW = None
        self.dB = None

    def forward(self, x):
        #テンソル対応
        self.original_x_shape = x.shape #元の形を記憶させる
        x = x.reshape(x.shape[0], -1)   #奥行き方向の幅を固定しつつ、行列の大きさを変更
        self.x = x
        
        out = np.dot(self.x, self.W) + self.B

        return out

    def backward(self, dout):
        #printの設定
        #print('第%d層 - AffineLayer - Weight%d, %d' %(counter, counter-1, counter))
        #print(self.W)
        #print('第%d層 - AffineLayer - Bias%d' %(counter, counter))
        #print(self.B)

        dx = np.dot(dout, self.W.T)
        #self.W = self.W - np.dot(self.x.T, dout)
        #self.B = self.B - np.sum(dout, axis = 0)
        self.W = np.dot(self.x.T, dout)
        self.B = np.sum(dout, axis = 0)

        dx = dx.reshape(*self.original_x_shape) #逆伝播を入力信号の形に戻す
        return dx

a1 = affine(W1, B1)
r1 = relu()
a2 = affine(W2, B2)
r2 = relu()
l = liner()
loss = mean_squared_error()

for i in range(20):
    x = a1.forward(x)
    x = r1.forward(x)
    x = a2.forward(x)
    x = r2.forward(x)
    x = l.forward(x)
    x = loss.forward(x, t)

    x = l.backward(x)
    x = r2.backward(x)
    x = a2.backward(x)
    x = r1.backward(x)
    x = a1.backward(x)


print(x)
