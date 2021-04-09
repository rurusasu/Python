import os
import sys
sys.path.append(".")
sys.path.append("..")

from multiprocessing import Pool
import urllib.request

from config.config import cfg
from src.utils import DownloadZip
"""
LINEMOD Dataset 製作者の Stefan Hinterstoißer さんのサイトから
LINEMOD Dataset の元データをダウンロードする。
"""

object_names = ['ape', 'benchviseblue', 'bowl', 'cam', 'can', 'cat',
                'cup', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                'iron', 'lamp', 'phone']

# Stefan Hinterstoißer さんのサイトのURL
base_url = 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/{}.zip'
target_dir = cfg.LINEMOD_DIR
temp_dir = cfg.TEMP_DIR

def main():
    for object_name in object_names:
        url = base_url.format(object_name)
        file_pth = DownloadZip(url, temp_dir)
        


if __name__ == "__main__":
    main()