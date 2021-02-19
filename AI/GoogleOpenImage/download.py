#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys

sys.path.append(".")
sys.path.append("..")

import pandas as pd
import requests

from src.config import cfg

# 画像ファイルをダウンロード
def Download_image(url, timeout=10):
    response = requests.get(url, allow_redirects=False)  # リダイレクト処理を無効
    if response.status_code != 200:  # HTTPステータスコードの200番台以外はエラーコード
        e = Exception("HTTP status: " + response.status_code)
        raise e

    content_type = response.headers["Content-Type"]
    if "image" not in content_type:
        e = Exception("content_type: " + content_type)
        raise e

    # 読み込んだURLを画面上に表示
    print(url)
    return response.content


# 画像のファイル名を決める
def Make_filename(base_dir, name, url):
    ext = os.path.splitext(url)[1]  # 拡張子の取得
    file_name = name + ext  # nameに拡張子をつける

    full_path = os.path.join(base_dir, file_name)
    return full_path


# 画像の保存
def Save_image(filename, image):
    with open(filename, "wb") as f:
        f.write(image)


class Download_GOI:
    ROOT_DIR = cfg.DATA_DIR + os.sep
    train_csv_path = ROOT_DIR + "train-images-boxable-with-rotation.csv"  # 訓練用CSVファイル

    # 保存先のファイル

    def Download_train(self, dst_path):
        f_path = self.train_csv_path
        # Original ImagesのURL部分を抽出
        df = pd.read_csv(f_path, usecols=["ImageID", "OriginalURL"], dtype=str)

        for id, url in df.values:
            file_name = Make_filename(dst_path, name=id, url=url)
            try:
                img = Download_image(url)
                Save_image(file_name, img)
            except:
                pass


if __name__ == "__main__":
    # f_path = cfg.DATA_DIR + os.sep + "train-images-boxable-with-rotation.csv"
    # 1つのURLを読み出し
    # df = pd.read_csv(f_path, usecols=["ImageID", "OriginalURL"], nrows=1, dtype=str)
    # print(df.values[0][0])
    # for id, url in df.values:
    #    print("id=" + id + "," + "url=" + url)
    # dst_path = cfg.TRAIN_DIR + os.sep + "test.jpg"
    # Download_file(df.values[0][0], dst_path)

    # すべてのURLを読み出し
    # df = pd.read_csv(f_path, usecols=["ImageID", "OriginalURL"], dtype=str)
    # print(df.shape[0])

    # url = df.values[0][0]
    # ext = os.path.splitext(url)  # ('https://farm3.staticflickr.com/5310/5898076654_51085e157c_o', '.jpg')
    # ext = os.path.splitext(url)[1]  # .jpg

    dst_path = cfg.TRAIN_DIR
    di = Download_GOI()
    di.Download_train(dst_path)