from easydict import EasyDict
import os
import sys

cfg = EasyDict()

"""
PathSettings
"""
cfg.LIB_DIR = os.path.dirname(os.path.abspath(__file__))
cfg.ROOT_DIR = os.path.dirname(cfg.LIB_DIR)
cfg.MODEL_DIR = os.path.join(cfg.ROOT_DIR, "model")
cfg.SRC_DIR = os.path.join(cfg.ROOT_DIR, "src")


def add_path():
    for key, value in cfg.items():
        if "DIR" in key:
            sys.path.insert(0, value)


add_path()

