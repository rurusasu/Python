import numpy as np

def dataLoad(filePath, dtype):
    if filePath != str:
        filePath = str(filePath)
    x = np.loadtxt(
        filePath,  # 読み込むファイル名(例"save_data.csv")
        dtype=dtype,  # データのtype
        delimiter=",",  # 区切り文字の指定
        ndmin=2  # 配列の最低次元
    )
    return x


file_path = (r'D:\myfile\My_programing\python\AI\NeuralNet\nn_pt6\data\training.csv')
#file_path = 'D:/myfile/My_programing/python/AI/NeuralNet/nn_pt6/data/training.csv'
x = dataLoad(file_path, float)
#x = np.loadtxt(file_path)
print(x)
