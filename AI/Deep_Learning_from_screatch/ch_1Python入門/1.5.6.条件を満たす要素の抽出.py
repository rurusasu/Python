import numpy as np
X = np.array([[51, 55], [14, 19], [0, 4]])
X = X.flatten()
print(X > 15)
print(X[X>15])