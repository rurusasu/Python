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

cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, "data")
cfg.EVAL_INDEXES_DIR = os.path.join(cfg.SRC_DIR, "eval_indexes")
cfg.GUI_DIR = os.path.join(cfg.ROOT_DIR, "gui")
cfg.NESTED_UNET_DIR = os.path.join(cfg.ROOT_DIR, "NestedUNet")

"""
DataDir
"""
cfg.TEST_DIR = os.path.join(cfg.DATA_DIR, "test")
cfg.BINARY_IMG_DIR = os.path.join(cfg.TEST_DIR, "binary_img")
cfg.CONTRAST_IMG_DIR = os.path.join(cfg.TEST_DIR, "contrast_img")
cfg.EDGE_IMG_DIR = os.path.join(cfg.TEST_DIR, "edge_img")
cfg.GRAY_IMG_DIR = os.path.join(cfg.TEST_DIR, "gray_img")
cfg.TEST_IMG_ORG_DIR = os.path.join(cfg.TEST_DIR, "org_img")
cfg.NOISE_IMG_DIR = os.path.join(cfg.TEST_DIR, "noise_img")
cfg.SMOOTHING_IMG_DIR = os.path.join(cfg.TEST_DIR, "smoothing_img")

"""cifar10
"""
cfg.CIFAR10_DIR = os.path.join(cfg.DATA_DIR, "cifar10")

"""DataScienceBowl
"""
cfg.DATA_SCIENCE_BOWL_DIR = os.path.join(cfg.DATA_DIR, "DataScienceBowl")
cfg.DSB2018_96_DIR = os.path.join(cfg.DATA_SCIENCE_BOWL_DIR, "dsb2018_96")

"""OpenImage
"""
cfg.OPENIMAGE_DIR = os.path.join(cfg.DATA_DIR, "OpenImage")
# Box Dir
cfg.BOX_DIR = os.path.join(cfg.OPENIMAGE_DIR, "box")
cfg.BOX_TRAIN_DIR = os.path.join(cfg.BOX_DIR, "train")
cfg.BOX_VALIDATION_DIR = os.path.join(cfg.BOX_DIR, "validation")
cfg.BOX_TEST_DIR = os.path.join(cfg.BOX_DIR, "test")
# Image id Dir
cfg.IMAGE_ID_DIR = os.path.join(cfg.OPENIMAGE_DIR, "image_id")
cfg.IMAGE_ID_TRAIN_DIR = os.path.join(cfg.IMAGE_ID_DIR, "train")
cfg.IMAGE_ID_VALIDATION_DIR = os.path.join(cfg.IMAGE_ID_DIR, "validation")
cfg.IMAGE_ID_TEST_DIR = os.path.join(cfg.IMAGE_ID_DIR, "test")
# Image label Dir
cfg.IMAGE_LABEL_DIR = os.path.join(cfg.OPENIMAGE_DIR, "image_label")
cfg.IMAGE_LABEL_TRAIN_DIR = os.path.join(cfg.IMAGE_LABEL_DIR, "train")
cfg.IMAGE_LABEL_VALIDATION_DIR = os.path.join(
    cfg.IMAGE_LABEL_DIR, "validation")
cfg.IMAGE_LABEL_TEST_DIR = os.path.join(cfg.IMAGE_LABEL_DIR, "test")
# Images Dir
cfg.IMAGES_DIR = os.path.join(cfg.OPENIMAGE_DIR, "images")
cfg.IMAGES_TRAIN_DIR = os.path.join(cfg.IMAGES_DIR, "train")
cfg.IMAGES_VALIDATION_DIR = os.path.join(cfg.IMAGES_DIR, "validation")
cfg.IMAGES_TEST_DIR = os.path.join(cfg.IMAGES_DIR, "test")
# Meta data Dir
cfg.META_DATA_DIR = os.path.join(cfg.OPENIMAGE_DIR, "meta_data")
cfg.META_DATA_TRAIN_DIR = os.path.join(cfg.META_DATA_DIR, "train")
cfg.META_DATA_VALIDATION_DIR = os.path.join(cfg.META_DATA_DIR, "validation")
cfg.META_DATA_TEST_DIR = os.path.join(cfg.META_DATA_DIR, "test")
# Relationship Dir
cfg.RELATIONSHIP_DIR = os.path.join(cfg.OPENIMAGE_DIR, "relationship")
cfg.RELATIONSHIP_TRAIN_DIR = os.path.join(cfg.RELATIONSHIP_DIR, "train")
cfg.RELATIONSHIP_VALIDATION_DIR = os.path.join(
    cfg.RELATIONSHIP_DIR, "validation")
cfg.RELATIONSHIP_TEST_DIR = os.path.join(cfg.RELATIONSHIP_DIR, "test")
# Segmentation Dir
cfg.SEGMENTATION_DIR = os.path.join(cfg.OPENIMAGE_DIR, "segmentation")
cfg.SEGMENTATION_TRAIN_DIR = os.path.join(cfg.SEGMENTATION_DIR, "train")
cfg.SEGMENTATION_VALIDATION_DIR = os.path.join(
    cfg.SEGMENTATION_DIR, "validation")
cfg.SEGMENTATION_TEST_DIR = os.path.join(cfg.SEGMENTATION_DIR, "test")

"""FashionMNIST
"""
cfg.FASHION_MNIST_DIR = os.path.join(cfg.DATA_DIR, "FashionMNIST")

"""画像データ
"""
cfg.IMAGE_DIR = os.path.join(cfg.DATA_DIR, "img")

"""Dobot
"""
cfg.DOBOT_DLL_DIR = os.path.join(cfg.GUI_DIR, "DobotDLL")
cfg.DOBOT_FUNCTION_DIR = os.path.join(cfg.GUI_DIR, "DobotFunction")

"""NestedUNet
"""
cfg.UNET_MODELS_DIR = os.path.join(cfg.NESTED_UNET_DIR, 'models')
cfg.UNET_RESULTS_DIR = os.path.join(cfg.NESTED_UNET_DIR, 'results')


def add_path():
    """システムのファイルパスを設定するための関数"""

    for key, value in cfg.items():
        if "DIR" in key:
            sys.path.insert(0, value)


add_path()
