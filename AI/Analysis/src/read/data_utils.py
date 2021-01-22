import os
import sys

sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch
from config.config import cfg
import glob
from pandas import plotting
import pandas as pd
from PIL import Image, ImageFile
from matplotlib import pyplot as plt


class DataUtils(object):
    def __init__(self):
        self.extensions = ['.png',
                           '.jpg',
                           '.bmp']

    def read_rgb_np(self, path: str, size: tuple = (64, 64), flatten=False) -> np.array:
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
        img = Image.open(path).convert('RGB')
        img = img.resize(size, Image.LANCZOS)
        img = np.array(img, np.uint8)
        if flatten:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            #img = img.reshape([1, -1, 3])
            img = np.column_stack([r.flatten(),
                                   g.flatten(),
                                   b.flatten()])
        return img

    def read_rgb(self, path: str) -> torch.Tensor:
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

        img = self.read_rgb_np(path)
        return torch.from_numpy(img).float().permute(2, 0, 1)

    def read_all_image(self, path: str, flatten: bool = False):
        """フォルダ内の画像、すべてを読み込み

        Param
        -----
        path (str):
            読み込む画像のパス
        """

        """
        img_path_list = []
        for ext in self.extensions:
            img_path_list.extend(glob.glob(path+os.sep+'**'+os.sep+'*'+ext, recursive=True))  # リストを結合する
        print(img_path_list)
        """
        img_data = {}
        img = {}
        #ImageData = pd.DataFrame()
        i = 1
        for ext in self.extensions:
            img_dir = path+os.sep+'**'+os.sep+'*'+ext
            for p in glob.iglob(img_dir, recursive=True):
                name = os.path.basename(os.path.dirname(p.rstrip(os.sep)))
                if i == 1:
                    name_memo = name

                if name != name_memo:
                    # 初期化処理
                    img = {}
                    i = 1
                    name_memo = name

                img[i] = self.read_rgb_np(p, flatten=flatten)
                img_data[name] = img

                i += 1

            if len(img_data) != 0:
                DataFrame = pd.DataFrame(img_data)
        return DataFrame

    def data_plot(self, df: pd.DataFrame):
        plotting.scatter_matrix(df.iloc[:, 1],
                                figsize=(8, 8),
                                c = list(df.iloc[:, 0]),alpha=0.5)
        plt.show()



if __name__ == "__main__":
    from config.config import cfg

    #angle_dir_path = cfg.ANGLE_ORIGINAL_DIR
    #dir_name = 'training'
    #base_dir = os.path.join(angle_dir_path, dir_name)

    base_dir = cfg.DEBUG_ONEIMAGE_DIR
    img_path = '0' + os.sep + 'image1.jpg'
    img_path = os.path.join(base_dir, img_path)

    utils = DataUtils()
    #img = utils.read_rgb_np(img_path, flatten=True)
    df = utils.read_all_image(base_dir)
    utils.data_plot(df)
