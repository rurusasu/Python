import sys, os

from easydict import EasyDict

cfg = EasyDict()


cfg.CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
cfg.ROOT_DIR = os.path.dirname(cfg.CONFIG_DIR)

"""
DataDir
"""
cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, "data")

def add_path():
    """
    システムのファイルパスを設定するための関数
    """

    for key, value in cfg.items():
        if "DIR" in key:
            sys.path.insert(0, value)


add_path()
