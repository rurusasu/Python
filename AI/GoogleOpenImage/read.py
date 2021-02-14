#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys

sys.path.append(".")
sys.path.append("..")

# from bs4 import BeautifulSoup
import pandas as pd
import requests

# import urllib3

from src.config import cfg


def download_file(url, dst_path):
    r = requests.get(url)
    with open(dst_path, "wb") as f:
        f.write(r.content)


if __name__ == "__main__":
    f_path = cfg.DATA_DIR + os.sep + "train-images-boxable-with-rotation.csv"
    df = pd.read_csv(f_path, usecols=["OriginalURL"], nrows=1, dtype=str)
    print(df.values[0][0])

    dst_path = cfg.TRAIN_DIR + os.sep + "test.jpg"
    download_file(df.values[0][0], dst_path)
