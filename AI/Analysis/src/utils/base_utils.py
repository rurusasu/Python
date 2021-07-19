import os
import sys
import json

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import numpy as np
import pickle
from PIL import Image, ImageFile
from plyfile import PlyData
from torch.utils.data import Sampler

from src.config.config import cfg


def read_mask_np(mask_path: str) -> np.ndarray:
    """
    マスク画像を読み出し，ndarray 配列 [max = 255，min = 0] として返す関数．

    Args:
        mask_path (str): マスク画像のパス

    Returns:
        mask_seg(np.ndarray): マスク画像の ndarray 配列
    """
    mask = Image.open(mask_path)
    mask_seg = np.array(mask).astype(np.int32)
    return mask_seg


def read_rgb_np(rgb_path: str) -> np.ndarray:
    """
    RGB画像を読み出し，ndarray 配列 [max = 255, min = 0] として返す関数

    Args:
        rgb_path(str): rgb画像のパス

    Returns:
        img (np.ndarray): rgb 画像の ndarray 配列
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(rgb_path).convert("RGB")
    img = np.array(img, np.uint8)
    return img


def read_pickle(pkl_path: str):
    """pickle データを読み出す関数

    Args:
        pkl_path (str): `.pkl` を含んだパス

    Returns:
        pkl_data:
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, pkl_path: str):
    """データを `.pkl` 形式で保存する関数

    Args:
        data (any): `.pkl` に保存するデータ
        pkl_path (str): データの保存先のパス
    """
    os.system("mkdir -p {}".format(os.path.dirname(pkl_path)))
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)


def read_ply_model(model_path: str) -> np.array:
    """
    `.ply` 形式で保存された 3D モデル(点群のx, y, z 座標)を numpy 配列として読み出す関数

    Arg:
        model_path(str): `.ply` 形式で保存された3Dモデルへのパス

    Return:
        (np.array): numpy配列に変換した 3D モデル
    """
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data["x"]
    y = data["y"]
    z = data["z"]
    return np.stack([x, y, z], axis=-1)


class Projector(object):
    intrinsic_matrix = {
        "linemod": np.array(
            [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
        ),
        "blender": np.array(
            [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        "pascal": np.asarray(
            [[-3000.0, 0.0, 0.0], [0.0, 3000.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    }

    def project(self, pts_3d, RT, K_type):
        pts_2d = np.matmul(pts_3d, RT[:, :3].T) + RT[:, 3:].T
        pts_2d = np.matmul(pts_2d, self.intrinsic_matrix[K_type].T)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d

    def project_h(self, pts_3dh, RT, K_type):
        """
        :param pts_3dh: [n,4]
        :param RT:      [3,4]
        :param K_type:
        :return: [n,3]
        """
        K = self.intrinsic_matrix[K_type]
        return np.matmul(np.matmul(pts_3dh, RT.transpose()), K.transpose())

    def project_pascal(self, pts_3d, RT, principle):
        """
        :param pts_3d:    [n,3]
        :param principle: [2,2]
        :return:
        """
        K = self.intrinsic_matrix["pascal"].copy()
        K[:2, 2] = principle
        cam_3d = np.matmul(pts_3d, RT[:, :3].T) + RT[:, 3:].T
        cam_3d[np.abs(cam_3d[:, 2]) < 1e-5, 2] = 1e-5  # revise depth
        pts_2d = np.matmul(cam_3d, K.T)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d, cam_3d

    def project_pascal_h(self, pts_3dh, RT, principle):
        K = self.intrinsic_matrix["pascal"].copy()
        K[:2, 2] = principle
        return np.matmul(np.matmul(pts_3dh, RT.transpose()), K.transpose())

    @staticmethod
    def project_K(pts_3d, RT, K):
        pts_2d = np.matmul(pts_3d, RT[:, :3].T) + RT[:, 3:].T
        pts_2d = np.matmul(pts_2d, K.T)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d


with open(os.path.join(cfg.CONFIG_DIR, "default_linemod_cfg.json"), "r",) as f:
    default_aug_cfg = json.load(f)


class ImageSizeBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last, cfg=default_aug_cfg):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of torch.utils.data.Sampler, but got sampler={}".format(
                    sampler
                )
            )
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integeral value, but got batch_size={}".format(
                    batch_size
                )
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got drop_last={}".format(
                    drop_last
                )
            )

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.hmin = cfg["hmin"]  # -> 256
        self.hmax = cfg["hmax"]  # -> 480
        self.wmin = cfg["wmin"]  # -> 256
        self.wmax = cfg["wmax"]  # -> 640
        self.size_int = cfg["size_int"]  # -> 8
        self.hint = (self.hmax - self.hmin) // self.size_int + 1  # -> 29
        self.wint = (self.wmax - self.wmin) // self.size_int + 1

    def generate_height_width(self):
        # hi, wi = np.random.randint(0, self.hint), np.random.randint(0, self.wint)
        # h, w = self.hmin + hi * self.size_int, self.wmin + wi * self.size_int
        h, w = self.hmin, self.wmin
        return h, w

    def __iter__(self):
        batch = []
        h, w = self.generate_height_width()
        for idx in self.sampler:
            batch.append((idx, h, w))
            if len(batch) == self.batch_size:
                h, w = self.generate_height_width()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            h, w = self.generate_height_width()
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
