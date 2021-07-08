import os
import sys

from numpy import ma

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import numpy as np
from PIL import Image, ImageFile
from plyfile import PlyData

from src.config.config import cfg


class LineModModelDB(object):
    """LineModModelDBは、各モデルのメッシュを管理するために使用されます

    """

    corners_3d = {}
    models = {}
    diameters = {}
    centers_3d = {}
    farthest_3d = {"8": {}, "4": {}, "12": {}, "16": {}, "20": {}}
    small_bbox_corners = {}

    def __init__(self):
        """LineModModelDB を初期化します.
        """

        self.linemod_dir = cfg.LINEMOD_DIR
        self.pvnet_linemod_dir = cfg.PVNET_LINEMOD_DIR

        self.ply_pattern = os.path.join(self.pvnet_linemod_dir, "{}/{}.ply")
        self.diameter_pattern = os.path.join(self.linemod_dir, "{}/distance.txt")
        self.farthest_pattern = os.path.join(
            self.pvnet_linemod_dir, "{}/farthest{}.txt"
        )

    def get_centers_3d(self, obj_name) -> np.ndarray:
        """
        単一のオブジェクトに対応する3D空間におけるバウンディングボックスの中心座標を返す関数

        Arg:
            obj_name (str): データを生成するLinemodオブジェクトのオブジェクト名.

        Return:
            centers_3d (np.ndarray): D空間におけるバウンディングボックスの中心座標.
        """
        if obj_name in self.centers_3d:
            return self.centers_3d[obj_name]

        c3d = self.get_corners_3d(obj_name)
        self.centers_3d[obj_name] = (np.max(c3d, 0) + np.min(c3d, 0)) / 2
        return self.centers_3d[obj_name]

    def get_corners_3d(self, obj_name: str) -> np.ndarray:
        """単一のオブジェクトに対応する3D空間におけるバウンディングボックスの8つの頂点座標の行列を返す関数

        Arg:
            obj_name (str): データを生成するLinemodオブジェクトのオブジェクト名.

        Return:
            corners_3d (np.ndarray): 3D空間におけるバウンディングボックスの8つの頂点座標の行列
        """
        if obj_name in self.corners_3d:
            return self.corners_3d[obj_name]

        corner_pth = os.path.join(self.pvnet_linemod_dir, obj_name, "corners.txt")
        if os.path.exists(corner_pth):
            # 'corners.txt' フォルダが存在する場合
            self.corners_3d[obj_name] = np.loadtxt(corner_pth)
            return self.corners_3d[obj_name]

        ply_pth = self.ply_pattern.format(obj_name, obj_name)
        ply = PlyData.read(ply_pth)
        data = ply.elements[0].data

        x = data["x"]
        min_x, max_x = np.min(x), np.max(x)
        y = data["y"]
        min_y, max_y = np.min(y), np.max(y)
        z = data["z"]
        min_z, max_z = np.min(z), np.max(z)
        corners_3d = np.array(
            [
                [min_x, min_y, min_z],
                [min_x, min_y, max_z],
                [min_x, max_y, min_z],
                [min_x, max_y, max_z],
                [max_x, min_y, min_z],
                [max_x, min_y, max_z],
                [max_x, max_y, min_z],
                [max_x, max_y, max_z],
            ]
        )
        np.savetxt(corner_pth, corners_3d)

        return corners_3d

    def get_diameter(self, obj_name):
        """直径を得る関数
        """
        if obj_name in self.diameters:
            return self.diameters[obj_name]

        diameter_path = self.diameter_pattern.format(obj_name)
        diameter = np.loadtxt(diameter_path) / 100.0
        self.diameters[obj_name] = diameter
        return diameter

    def get_farthest_3d(self, obj_name: str, num: int = 8) -> np.ndarray:
        """
        farthest で

        Args:
            obj_name (str): [description]
            num (int, optional): [description]. Defaults to 8.

        Returns:
            np.ndarray: [description]
        """
        if obj_name in self.farthest_3d["{}".format(num)]:
            return self.farthest_3d["{}".format(num)][obj_name]

        if num == 8:
            farthest_path = self.farthest_pattern.format(obj_name, "")
        else:
            farthest_path = self.farthest_pattern.format(obj_name, num)
        farthest_pts = np.loadtxt(farthest_path)
        self.farthest_3d["{}".format(num)][obj_name] = farthest_pts
        return farthest_pts

    def get_ply_mesh(self, obj_name):
        ply = PlyData.read(self.ply_pattern.format(obj_name, obj_name))
        vert = np.asarray(
            [ply["vertex"].data["x"], ply["vertex"].data["y"], ply["vertex"].data["z"]]
        ).transpose()
        vert_id = [id for id in ply["face"].data["vertex_indices"]]
        vert_id = np.asarray(vert_id, np.int64)

        return vert, vert_id

    def get_ply_model(self, obj_name):
        if obj_name in self.models:
            return self.models[obj_name]

        ply = PlyData.read(self.ply_pattern.format(obj_name, obj_name))
        data = ply.elements[0].data
        x = data["x"]
        y = data["y"]
        z = data["z"]
        model = np.stack([x, y, z], axis=-1)
        self.models[obj_name] = model
        return model

    def get_small_bbox(self, obj_name):
        if obj_name in self.small_bbox_corners:
            return self.small_bbox_corners[obj_name]

        corners = self.get_corners_3d(obj_name)
        center = np.mean(corners, 0)
        small_bbox_corners = (corners - center[None, :]) * 2.0 / 3.0 + center[None, :]
        self.small_bbox_corners[obj_name] = small_bbox_corners

        return small_bbox_corners


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    obj_name = "ape"
    db = LineModModelDB()

    # --------------------- #
    # corners_3d Test #
    # --------------------- #
    corners_3d = db.get_corners_3d(obj_name=obj_name)
    print("corners_3d = ")
    print(corners_3d)
    print("shape =", corners_3d.shape)

    # --------------------------- #
    # small_bbox_corners #
    # --------------------------- #
    small_bbox_corners = db.get_small_bbox(obj_name=obj_name)
    print("small_bbox_corners = ")
    print(small_bbox_corners)
    print("shape =", small_bbox_corners.shape)

    # -------------- #
    # ply_mesh  #
    # -------------- #
    ply_mesh = db.get_ply_mesh(obj_name=obj_name)
    print("ply_mesh= ")
    print(ply_mesh)
    print("len =", len(ply_mesh))
    print("shape =", ply_mesh[0].shape)
    print("shape =", ply_mesh[1].shape)

