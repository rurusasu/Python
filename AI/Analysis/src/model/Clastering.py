import os
import sys

sys.path.append('.')
sys.path.append('..')

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt


class cluster(object):

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def k_mean(self, features: np.array):

        # モデルの作成
        model = KMeans(n_clusters=36).fit(features)

        #makers=['+', '*', 'o']
        #color=['r', 'b', 'g']
        for i in range(36):
            f = features[model.labels_ ==i, :]
            plt.scatter(f[:,0],f[:,1])

        plt.show()
        """
        # クラスタ数を変更して試したいので古い出力結果は消す
        for i in range(model.n_clusters):
            cluster_dir = OUTPUT_DIR + os.sep + "cluster{}".format(i)
            if os.path.exists(cluster_dir):
                shutil.rmtree(cluster_dir)
            os.makedirs(cluster_dir)

        # 結果をクラスタ毎にディレクトリに保存
        for label, p in zip(model.labels_, im)
        """

    def __get_value__(self):
        value = self.data.values[0]
        return value

    def __flatten__(self, value: np.array):
        """ numpy arrayを1行のデータに変換する
        例：
            img = [['Red'],
                   ['Green'],
                   ['Blue']]
            の2次元配列があるとき、これらの値を
            flat_data = ['Red', 'Green', 'Blue']
            の1行配列に変換する。
        """
        num = 1
        for i in value:
            flat_img = i.flatten()  # 2次元配列 -> 1次元配列
            if num == 1:
                flat_data = flat_img
            else:
                # 行方向にスタック
                flat_data = np.column_stack([flat_data, flat_img])
            num += 1

        return flat_data


if __name__ == "__main__":
    from config.config import cfg
    from src.read.data_utils import DataUtils

    base_dir = cfg.DEBUG_ONEIMAGE_DIR

    utils = DataUtils()
    df = utils.read_all_image(base_dir, flatten=True)

    cls = cluster(data=df)
    df = cls.__get_value__()
    flat_data = cls.__flatten__(df)
    model = cls.k_mean(flat_data)
