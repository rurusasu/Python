# cording: utf-8

import matplotlib.pyplot as plt
import numpy as np


class sgd:

    """確率的勾配降下法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class momentum_sgd:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr # 学習率
        self.momentum = momentum # 運動量
        self.v = None # 速度変化の項

    def update(self, params, grads):
        # 最適化1回目
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - \
                (1-self.momentum)*self.lr*grads[key]
            params[key] += self.v[key]


class nag:

    """ネステロフの加速度法（Nesterov's Accelerated Gradient method）"""
    
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr  # 学習率
        self.momentum = momentum  # 運動量
        self.v = None  # 速度変化の項

    def update(self, params, grads):
        # 最適化1回目
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.momentum*self.momentum*self.v[key]
            params[key] -= (1-self.momentum)*self.lr*grads[key]


class ada_grad:

    """Adaptive subGradient descent"""

    def __init__(self, lr=0.001):
        self.lr = lr # 学習率
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)  


class rmsprop:

    """RSMprop"""

    def __init__(self, lr=0.01, rho=0.99, eps=1e-6):
        self.lr = lr  # 学習率
        self.rho = rho
        self.h = None
        self.eps = eps

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] *= self.rho
            self.h[key] += (1 - self.rho) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.eps)



class ada_delta:

    """Adaptive subGradient delta"""

    def __init__(self, lr, rho=0.95):
        self.rho = rho
        self.h = None
        self.u = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            self.u = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                self.u[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] *= self.rho
            self.h[key] += (1 - self.rho) * grads[key] * grads[key]
            dx = np.sqrt((self.u[key] + 1e-7) / (self.h[key] + 1e-7)) * grads[key]
            self.u[key] *= self.rho
            self.u[key] += (1 - self.rho) * dx * dx
            params[key] -= dx


class adam:
    def __init__(self, lr=0.001, rho1=0.9, rho2=0.999, eps=1e-8):
        self.lr = lr
        self.rho1 = rho1
        self.rho2 = rho2
        self.eps = eps
        self.m = None
        self.v = None
    
    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.m[key] += (1 - self.rho1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.rho2) * (grads[key] * grads[key] - self.v[key])
            params[key] -= self.lr * self.m[key] / (np.sqrt(self.v[key]) + self.eps)




if __name__ == "__main__":
    from collections import OrderedDict
    import matplotlib.pyplot as plt


    def f(x, y):
        return x**2 / 20.0 + y**2

    def df(x, y):
        return x / 10.0, 2.0*y

    init_pos = (-7.0, 2.0)
    params = {}
    grads = {}
    grads['x'], grads['y'] = 0, 0

    optimizers = OrderedDict()
    optimizers['SGD'] = sgd(lr=0.95)
    optimizers['momentum_SGD'] = momentum_sgd(lr=0.01)

    idx = 1

    for key in optimizers:
        optimizer = optimizers[key]
        x_history = []
        y_history = []
        params['x'], params['y'] = init_pos[0], init_pos[1]

        for i in range(30):
            x_history.append(params['x'])
            y_history.append(params['y'])

            grads['x'], grads['y'] = df(params['x'], params['y'])
            optimizer.update(params, grads)

        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)

        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        # for simple contour line
        mask = Z > 7
        Z[mask] = 0

        # plot
        #plt.subplot(2, 2, idx)
        idx += 1
        plt.plot(x_history, y_history, 'o-', color="red")
        plt.contour(X, Y, Z)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, '+')
        #colorbar()
        #spring()
        plt.title(key)
        plt.xlabel("x")
        plt.ylabel("y")

    plt.show()
