import os
import sys

from easydict import EasyDict

cfg = EasyDict()

"""Path Setting
"""

cfg.SRC_DIR = os.path.dirname(os.path.abspath(__file__))
cfg.ROOT_DIR = os.path.dirname(cfg.SRC_DIR)

"""DataDir
"""

cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, "Data")
cfg.TRAIN_DIR = os.path.join(cfg.DATA_DIR, "train")


def add_path():
    """システムのファイルパスを設定するための関数"""

    for key, value in cfg.items():
        if "DIR" in key:
            sys.path.insert(0, value)


add_path()
