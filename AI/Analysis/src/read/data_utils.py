import os
import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch

import glob
import pandas as pd
from pandas import plotting
from matplotlib import pyplot as plt
from PIL import Image, ImageFile

from config.config import cfg


def (read_rgb_nppath: str, size: tuple = (64, 64), flatten=False) -> np.array:
    """画像を読み込み ndarray に変換する関数

    Param
    -----
    path (str):
        読み込む画像のパス

    Return
    ------
    img (ndarray):
        uint 8 の numpy 配列
    """

    # PIL は極端に大きな画像など高速にロードできない画像は見過ごす仕様になっている。
    # `LOAD_TRUNCATED_IMAGES` を `True` に設定することで、きちんとロードされるようになる。
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    img = np.array(img, np.uint8)
    if flatten:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        # img = img.reshape([1, -1, 3])
        img = np.column_stack([r.flatten(), g.flatten(), b.flatten()])
    return img


def read_rgb(path: str) -> torch.Tensor:
    """画像を読み込み torch の tensor に変換する関数

    Param
    -----
    path (str):
        読み込む画像のパス

    Return
    ------
    torch_tensor (tensor):
        float 32 のテンソル
    """

    img = read_rgb_np(path)
    return torch.from_numpy(img).float().permute(2, 0, 1)


class DataUtils(object):
    def __init__(self, load_dir_path: str):
        search_path = load_dir_path + os.sep + "**" + os.sep + ".csv"
        csv_path = glob.glob(search_path)
        for f_path in csv_path:
            if "train" in f_path:
                self.train_csv_path = f_path
            elif "validation" in f_path:
                self.validation_csv_path = f_path
            elif "test" in f_path:
                self.test_csv_path = f_path
        self.extensions = [".png", ".jpg", ".bmp"]


    def read_all_image(self, csv_path = self.train_csv_path, minibatch: int=30, flatten: bool = False):
        """フォルダ内の画像、すべてを読み込むための関数

        Param
        -----
        path (str):
            読み込む画像のパス

        Return
        ------
        Dataframe
        """

        """
        img_path_list = []
        for ext in self.extensions:
            img_path_list.extend(glob.glob(path+os.sep+'**'+os.sep+'*'+ext, recursive=True))  # リストを結合する
        print(img_path_list)
        """
        ext = ".jpg"
        img_data = {}
        img = {}
        # ImageData = pd.DataFrame()

        df = pd.DataFrame(csv_path, usecols=["ImageID"], dtype=str)
        size = df.size[0]

        # データをサンプリングするための乱数を発生させる
        np.random.seed(100) # シードを度定する
        id = np.random.randint(1, size, size=minibatch)

        for i in id
            labels = df.at[df.index[i], 'ImageID']
            # ロードするファイルのパスを指定
            f_path = load_dir_path + os.sep + labels + ext
            img[labels] = read_rgb_np(f_path, flatten=flatten)

        return pd.DataFrame(img)


    def data_plot(self, df: pd.DataFrame):
        """
        DataFrameに格納されている画像を表示するための関数

        Param
        -----
        df (DataFrame):
            表示するデータが格納されているデータフレーム
        """

        if df.size != 0:
            # データフレームにデータが格納されている場合
            plotting.scatter_matrix(
                df.iloc[:, 1], figsize=(8, 8), c=list(df.iloc[:, 0]), alpha=0.5
            )
            plt.show()
        else:
            # 格納されていない場合
            print("表示するデータがありません．")


if __name__ == "__main__":
    from config.config import cfg

    base_dir = cfg.DEBUG_ONEIMAGE_DIR
    img_path = "0" + os.sep + "image1.jpg"
    img_path = os.path.join(base_dir, img_path)

    utils = DataUtils()
    # img = utils.read_rgb_np(img_path, flatten=True)
    df = utils.read_all_image(base_dir)
    utils.data_plot(df)
