import os
import sys
sys.path.append(".")
sys.path.append("..")

from config.config import cfg
from src.utils import DownloadZip, UnpackZip
"""
LINEMOD Dataset 製作者の Stefan Hinterstoißer さんのサイトから
LINEMOD Dataset の元データをダウンロードする。
"""

"""
object_names = ['ape', 'benchviseblue', 'bowl', 'cam', 'can', 'cat',
                'cup', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                'iron', 'lamp', 'phone']
"""
object_names = ['phone']

# Stefan Hinterstoißer さんのサイトのURL
base_url = 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/{}.zip'
target_dir = cfg.LINEMOD_DIR
temp_dir = cfg.TEMP_DIR


def main():
    for object_name in object_names:
        url = base_url.format(object_name)
        f_path = DownloadZip(url, temp_dir)

        unpach_path = os.path.join(cfg.LINEMOD_DIR, object_name)
        UnpackZip(f_path, unpach_path)


if __name__ == "__main__":
    main()