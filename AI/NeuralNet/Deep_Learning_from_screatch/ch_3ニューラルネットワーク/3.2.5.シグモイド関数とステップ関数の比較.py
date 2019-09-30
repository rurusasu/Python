import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid_function(x)

plt.plot(x, y1, linestyle = "--", label="step_function")
plt.plot(x, y2, label="sigmoid_function")
plt.xlabel("x")
plt.ylabel("y")
plt.title('step & sigmoid')
plt.legend()
plt.show()