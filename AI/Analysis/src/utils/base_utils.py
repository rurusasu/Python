import os

import numpy as np
import pickle
from PIL import Image, ImageFile
from plyfile import PlyData


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
