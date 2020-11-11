from easydict import EasyDict
import os
import sys

cfg = EasyDict()

"""
Path Setting
"""

cfg.CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
cfg.SRC_DIR = os.path.dirname(cfg.CONFIG_DIR)
cfg.ROOT_DIR = os.path.dirname(cfg.SRC_DIR)
cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, 'data')
cfg.GUI_DIR = os.path.join(cfg.ROOT_DIR, 'gui')
cfg.EVAL_INDEXES_DIR = os.path.join(cfg.SRC_DIR, 'eval_indexes')

"""
DataDir
"""
cfg.TEST_DIR = os.path.join(cfg.DATA_DIR, 'test')


def add_path():
    """システムのファイルパスを設定するための関数
    """

    for key, value in cfg.items():
        if 'DIR' in key:
            sys.path.insert(0, value)


add_path()
