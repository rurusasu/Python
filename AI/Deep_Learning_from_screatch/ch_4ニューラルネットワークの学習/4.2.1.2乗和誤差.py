from pathlib import Path
import sys
sys.path.append(Path.cwd().parent)
import numpy as np
from common.functions import mean_squared_error

#「2」を正解とする
t = [0, 0, 1, 0, 0, 0, 0, 0, 0,0]

#例1:「2」の確率が最も高い場合（0.6）
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
mean_squared_error(np.array(y1), np.array(t))

#例2:「7」の確率が最も高い場合（0.6）
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mean_squared_error(np.array(y2), np.array(t))
