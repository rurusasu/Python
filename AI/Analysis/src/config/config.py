import os
import sys

from easydict import EasyDict

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

"""
Angledetection
"""
cfg.ANGLE_DETECTION_DIR = os.path.join(cfg.DATA_DIR, 'AngleDetection')
#################
# DeBug用
#################
cfg.DEBUG_DIR = os.path.join(cfg.ANGLE_DETECTION_DIR, 'DeBug')
cfg.DEBUG_ONEIMAGE_DIR = os.path.join(cfg.DEBUG_DIR, 'OneImage')
#################
# OrigData
#################
cfg.ANGLE_ORIGINAL_DIR = os.path.join(cfg.ANGLE_DETECTION_DIR, 'Original')

#################
# TrainingData_1
#################
cfg.ANGLE_TRAININGDATA1_DIR = os.path.join(
    cfg.ANGLE_DETECTION_DIR, 'TrainingData_1')
cfg.ANGLE_TRAININGDATA1_TRAINING_DIR = os.path.join(
    cfg.ANGLE_TRAININGDATA1_DIR, 'training')
cfg.ANGLE_TRAININGDATA1_VALIDATION_DIR = os.path.join(
    cfg.ANGLE_TRAININGDATA1_DIR, 'validation')

#################
# TrainingData_2
#################
cfg.ANGLE_TRAININGDATA2_DIR = os.path.join(
    cfg.ANGLE_DETECTION_DIR, 'TrainingData_2')


def add_path():
    """システムのファイルパスを設定するための関数
    """

    for key, value in cfg.items():
        if 'DIR' in key:
            sys.path.insert(0, value)


add_path()
