from easydict import EasyDict
import os
import sys

cfg = EasyDict()

"""PathSettings
"""

cfg.LIB_DIR = os.path.dirname(os.path.abspath(__file__))
cfg.ROOT_DIR = os.path.dirname(cfg.LIB_DIR)
cfg.DATASETS_DIR = os.path.join(cfg.ROOT_DIR, "datasets")
cfg.MODEL_DIR = os.path.join(cfg.ROOT_DIR, "model")
cfg.SRC_DIR = os.path.join(cfg.ROOT_DIR, "src")


def add_path():
    for key, value in cfg.items():
        if "DIR" in key:
            sys.path.insert(0, value)


add_path()


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="iris",
        choices=[
            "ilis",
            "boston",
            "diabetes",
            "digits",
            "linnerrud",
            "wine",
            "breast_cancer",
        ],
    )
    parser.add_argument(
        "--network", type=str, default="perseptron", choices=["perseptron", "adaline"]
    )
    parser.add_argument("--optimizer", type=str, default="GD", choices=["GD", "SGD"])
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--iter", type=int, default=10)
    args = parser.parse_args()

    return args
