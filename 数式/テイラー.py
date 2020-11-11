import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-1, 1, 0.1) # x=[-1.0, -0.9, ... 1.0]
y = np.exp(x)  # y = e^x
y_diff_1 = 1 + x
y_diff_2 = y_diff_1 + (1/2) * x**2
y_diff_3 = y_diff_2 + (1/6) * x**3


# Figureの初期化
fig = plt.figure(figsize=(8, 8))

plt.plot(x, y, color="b", marker="o", label="e^x")
plt.plot(x, y_diff_1, color="g", ls="-.", label="1+x")
plt.plot(x, y_diff_2, color="r", ls="--", label="1+x+(1/2)x^2")
plt.plot(x, y_diff_3, color="k", ls=":", label="1+x+(1/2)x^2+(1/6)x^3")

# 軸ラベルを設定
plt.xlabel("x", fontsize=18)
plt.ylabel("y", fontsize=18)
# X軸の最大値、最小値を指定
plt.xlim(x.min(), x.max()) 
# タイトルを表示
plt.title('Maclaurin(e^x)', fontsize=25)
# 凡例を表示
plt.legend()


plt.show()