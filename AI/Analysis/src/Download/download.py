#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import traceback
import urllib
import zipfile
from typing import Type, Union


sys.path.append(".")
sys.path.append("..")

import pandas as pd
import requests

from config.config import cfg
from src.utils.base_utils import MakeDir


# 画像ファイルをダウンロード
def Download_image(url: str, timeout: int = 10):
    """URL 先の HP で動作している HTML テキスト内に含まれる画像情報を検出し，ダウンロードするための関数

    Args:
        url (str): ダウンロードしたい画像が掲載されているサイトの URL
        timeout (int, optional): タイムアウト. Defaults to 10[s].

    Raises:
        e: [description]
        e: [description]

    Returns:
        [type]: [description]
    """
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
def Make_filename(base_dir: str, name: str, url: str):
    ext = os.path.splitext(url)[1]  # 拡張子の取得
    file_name = name + ext  # nameに拡張子をつける

    full_path = os.path.join(base_dir, file_name)
    return full_path


# 画像の保存
def Save_image(filename: str, image):
    with open(filename, "wb") as f:
        f.write(image)


def Download_images_from_csv(csv_path, dst_path):
    # Original ImagesのURL部分を抽出
    df = pd.read_csv(csv_path, usecols=["ImageID", "OriginalURL"], dtype=str)

    for id, url in df.values:
        file_name = Make_filename(dst_path, name=id, url=url)
        try:
            img = Download_image(url)
            Save_image(file_name, img)
        except:
            pass


class Download_GOI(object):
    """Google Open Image Dataset を自動でダウンロードするためのクラス"""

    # ROOT_DIR = cfg.DATA_DIR + os.sep
    # train_csv_path = ROOT_DIR + "train-images-boxable-with-rotation.csv"  # 訓練用CSVファイル
    csv_path = (
        cfg.IMAGE_ID_TRAIN_DIR + os.sep + "train-images-boxable-with-rotation.csv",
        cfg.IMAGE_ID_VALIDATION_DIR + os.sep + "validation-images-with-rotation.csv",
        cfg.IMAGE_ID_TEST_DIR + os.sep + "test-images-with-rotation.csv",
    )  # 訓練用CSVファイル

    train_images_dir = cfg.IMAGES_TRAIN_DIR
    validation_images_dir = cfg.IMAGES_VALIDATION_DIR
    test_images_dir = cfg.IMAGES_TEST_DIR

    def Download_train_images(self, csv_path=csv_path[0], dst_path=train_images_dir):
        Download_images_from_csv(csv_path, dst_path)

    def Download_validation_images(
        self, csv_path=csv_path[1], dst_path=validation_images_dir
    ):
        Download_images_from_csv(csv_path, dst_path)

    def Download_test_images(self, csv_path=csv_path[2], dst_path=test_images_dir):
        Download_images_from_csv(csv_path, dst_path)


def Download_ZIP(file_url: str, save_file_pth: str):
    """url 上にある zip ファイルを拡張子 .zip ファイルに保存するための関数

    Args:
        file_url (str): ダウンロードしたいファイルの url
        save_file_pth (str): `.zip` ファイルで終了するファイルパス。例: "/home/data/xxx.zip"
    """
    try:
        with urllib.request.urlopen(file_url) as f:
            data = f.read()
            with open(save_file_pth, mode="wb") as f_save:
                f_save.write(data)
                f_save.flush()
    except urllib.error.URLError as e:
        raise Exception("ファイルダウンロード時にエラーが発生しました。")
        #print(e)


def Unpack(file_pth: str, save_pth: str, file_name: Type[Union[str, None]] = None):
    """originalディレクトリ内のzipファイルを展開するための関数
    Arg:
        file_name(str): original ディレクトリ内の zip ファイル名,
        create_dir(bool optional): 解凍するときに、解凍前のファイル名と同じ名前のディレクトリを作成する。default to True.
    """
    with zipfile.ZipFile(file_pth) as obj_zip:
        if file_name != None:
            # file_name で指定したファイルだけを保存したい場合
            # zipから指定ファイル（第1引数）を取得して、指定ディレクトリ（第2引数）に保存する
            obj_zip.extract(file_name, save_pth)
        else:
            # zip に保存されているデータをすべて展開する
            obj_zip.extractall(save_pth)


class Download_LineMOD(object):
    def __init__(self, output_dir: str):
        # 保存先のディレクトリ作成
        self.file_url = "https://zjueducn-my.sharepoint.com/LINEMOD.tar.gz"
        self.output_pth = MakeDir(output_dir, newly=True)

    def download(self):
        print("ファイルのダウンロードを開始します。")
        try:
            Download_ZIP(file_url=self.file_url, save_file_pth=self.output_pth)
        except Exception as e:
            print(traceback.format_exc())
        else:
            print("ファイルを展開します。")
            Unpack(file_pth=self.output_pth, save_pth=self.output_pth)


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

    # dst_path = cfg.TRAIN_DIR
    # di = Download_GOI()
    # di.Download_validation_images()

    file_pth = os.path.join(cfg.PVNET_LINEMOD_DIR, "linemod.zip")
    Linemod = Download_LineMOD(file_pth)
    Linemod.download()