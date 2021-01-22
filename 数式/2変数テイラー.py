import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

a, b = 1, 1

x1 = np.arange(-1, 1.1, 0.1)  # x=[-1.0, -0.9, ... 1.0]
x2 = np.arange(-1, 1.1, 0.1)  # y=[-1.0, -0.9, ... 1.0]

x1, x2 = np.meshgrid(x1, x2)
#x = np.c_[np.ravel(x1), np.ravel(x2)]


z = np.exp(x1+(2*x2))  # z = e^(x+2y)
z_diff_1 = (1 + x1 + 2 * x2)
z_diff_2 = z_diff_1 + (x1**2 / 2) + 2*x1*x2 + 2*x2**2
z_diff_3 = z_diff_2 + 

#z_diff_1 = 1 + h + 2k
#y_diff_2 = y_diff_1 + (1/2) * x**2
#y_diff_3 = y_diff_2 + (1/6) * x**3


# Figureの初期化
fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x1, x2, z, cmap='bwr', linewidth=0, label='e^(x+2y)')
surf_diff_1 = ax.plot_surface(x1, x2, z_diff_1, linewidth=0, label='1+x+2y')
surf_diff_2 = ax.plot_surface(x1, x2, z_diff_2, linewidth=0, label='1+x+2y+x^2/2 + 2xy + 2y^2')

#plt.plot(x, y, color="b", marker="o", label="e^x")
#plt.plot(x, y_diff_1, color="g", ls="-.", label="1+x")
#plt.plot(x, y_diff_2, color="r", ls="--", label="1+x+(1/2)x^2")
#plt.plot(x, y_diff_3, color="k", ls=":", label="1+x+(1/2)x^2+(1/6)x^3")

# 軸ラベルを設定
ax.set_xlabel("x", fontsize=18)
ax.set_ylabel("y", fontsize=18)
# X軸Y軸の最大値、最小値を指定
ax.set_xlim(x1.min(), x1.max()) 
ax.set_ylim(x2.min(), x2.max())
# タイトルを表示
ax.set_title('Maclaurin(e^x)', fontsize=25)
# 凡例を表示
#ax.legend()


plt.show()