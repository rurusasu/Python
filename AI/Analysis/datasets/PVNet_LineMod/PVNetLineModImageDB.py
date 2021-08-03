import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import numpy as np
from skimage.io import imread

from datasets.Blender.render_base_utils import PoseTransformer
from datasets.LineMod.LineModDB import read_pose
from datasets.PVNet_LineMod.PVNetLineModModelDB import PVNetLineModModelDB
from src.config.config import cfg
from src.utils.base_utils import read_pickle, save_pickle, Projector


class PVNetLineModImageDB(object):
    """
    rgb_pth relative path to cfg.LINEMOD
    dpt_pth relative path to cfg.LINEMOD
    RT np.float32 [3,4]
    object_typ 'cat' ...
    rnd_typ 'real' or 'render'
    corner  np.float32 [8,2]
    """

    def __init__(
        self,
        obj_name: str,
        render_num: int = cfg.NUM_SYN,
        fuse_num: int = cfg.FUSE_NUM,
        ms_num: int = 10000,
        has_render_set=True,
        has_fuse_set=True,
    ):
        """LINEMOD が提供しているデータと Blender および fuse を使用して拡張したデータセットを管理するクラス

        Args:
            obj_name (str): オブジェクト名
            render_num (int, optional): Blender で作成したデータからロードするデータ数. Defaults to cfg.NUM_SYN.
            fuse_num (int, optional): fuse で作成したデータからロードするデータ数. Defaults to cfg.FUSE_NUM.
            ms_num (int, optional): [description]. Defaults to 10000.
            has_render_set (bool, optional): Blender で作成したデータをロードするか. Defaults to True.
            has_fuse_set (bool, optional): fuse で作成したデータをロードするか. Defaults to True.
        """

        self.linemod_dir = cfg.LINEMOD_DIR
        self.pvnet_linemod_dir = cfg.PVNET_LINEMOD_DIR
        #self.pvnet_linemod_dir = cfg.TEMP_DIR

        self.obj_name = obj_name
        # some dirs for processing
        # os.path.join(
        #    self.pvnet_linemod_dir, "posedb", "{}_render.pkl".format(obj_name)
        # )
        self.render_dir = "renders/{}".format(obj_name)
        self.rgb_dir = "{}/JPEGImages".format(obj_name)
        self.mask_dir = "{}/mask".format(obj_name)
        self.rt_dir = os.path.join(self.linemod_dir, obj_name, "data")
        self.render_num = render_num

        self.test_fn = "{}/test.txt".format(obj_name)
        self.train_fn = "{}/train.txt".format(obj_name)
        self.val_fn = "{}/val.txt".format(obj_name)

        # render で拡張したデータを読み込み
        if has_render_set:
            self.render_pkl = os.path.join(
                self.pvnet_linemod_dir, "posedb", "{}_render.pkl".format(obj_name)
            )
            # prepare dataset
            if os.path.exists(self.render_pkl):
                # read cached
                self.render_set = read_pickle(self.render_pkl)
            else:
                # process render set
                self.render_set = self.collect_render_set_info(
                    self.render_pkl, self.render_dir
                )
        else:
            self.render_set = []

        self.real_pkl = os.path.join(
            self.pvnet_linemod_dir, "posedb", "{}_real.pkl".format(obj_name)
        )
        if os.path.exists(self.real_pkl):
            # read cached
            self.real_set = read_pickle(self.real_pkl)
        else:
            # process real set
            self.real_set = self.collect_real_set_info()

        # prepare train test split
        self.train_real_set = []
        self.test_real_set = []
        self.val_real_set = []
        self.collect_train_val_test_info()

        self.fuse_set = []
        self.fuse_dir = "fuse"
        self.fuse_num = fuse_num
        self.obj_idx = cfg.linemod_obj_names.index(obj_name)

        if has_fuse_set:
            self.fuse_pkl = os.path.join(
                self.pvnet_linemod_dir, "posedb", "{}_fuse.pkl".format(obj_name)
            )
            # prepare dataset
            if os.path.exists(self.fuse_pkl):
                # read cached
                self.fuse_set = read_pickle(self.fuse_pkl)
            else:
                # process render set
                self.fuse_set = self.collect_fuse_info()
        else:
            self.fuse_set = []

    def collect_render_set_info(
        self, pkl_file: str, render_dir: str, format: str = "jpg"
    ) -> list:
        """
        `render_utils.py` で作成した `JPEG_IMAGE`, `depth_IMAGE`, `RT.pkl` に加え，`Bounding boxの頂点座標および中心座標`などを`database`配列にまとめて `pkl` ファイルに保存する関数．返り値は `database` 配列

        Args:
            pkl_file (str): render_utils で作成した `RT.pkl` のファイルパス
            render_dir (str): render_utils で作成した `renders` ディへクトリへのパス
            format (str, optional): 読み出すrgb画像のフォーマット. Defaults to "jpg".

        Returns:
            database (list): レンダリングしたオブジェクトに対する画像や回転行列などの情報をまとめた配列
        """
        database = []
        projector = Projector()
        modeldb = PVNetLineModModelDB()
        for k in range(self.render_num):
            data = {}
            data["rgb_pth"] = os.path.join(render_dir, "{}.{}".format(k, format))
            data["dpt_pth"] = os.path.join(render_dir, "{}_depth.png".format(k))
            data["RT"] = read_pickle(
                os.path.join(self.pvnet_linemod_dir, render_dir, "{}_RT.pkl".format(k))
            )["RT"]
            data["object_typ"] = self.obj_name
            data["rnd_typ"] = "render"
            data["corners"] = projector.project(
                modeldb.get_corners_3d(self.obj_name), data["RT"], "blender"
            )
            data["farthest"] = projector.project(
                modeldb.get_farthest_3d(self.obj_name), data["RT"], "blender"
            )
            data["center"] = projector.project(
                modeldb.get_centers_3d(self.obj_name)[None, :], data["RT"], "blender"
            )
            for num in [4, 12, 16, 20]:
                data["farthest{}".format(num)] = projector.project(
                    modeldb.get_farthest_3d(self.obj_name, num), data["RT"], "blender",
                )
            data["small_bbox"] = projector.project(
                modeldb.get_small_bbox(self.obj_name), data["RT"], "blender"
            )
            axis_direct = np.concatenate([np.identity(3), np.zeros([3, 1])], 1).astype(
                np.float32
            )
            data["van_pts"] = projector.project_h(axis_direct, data["RT"], "blender")
            database.append(data)

        save_pickle(database, pkl_file)
        return database

    def collect_real_set_info(self):
        database = []
        projector = Projector()
        modeldb = PVNetLineModModelDB()
        img_num = len(os.listdir(os.path.join(self.pvnet_linemod_dir, self.rgb_dir)))
        for k in range(img_num):
            data = {}
            data["rgb_pth"] = os.path.join(self.rgb_dir, "{:06}.jpg".format(k))
            data["dpt_pth"] = os.path.join(self.mask_dir, "{:04}.png".format(k))
            pose = read_pose(
                os.path.join(self.rt_dir, "rot{}.rot".format(k)),
                os.path.join(self.rt_dir, "tra{}.tra".format(k)),
            )
            pose_transformer = PoseTransformer(
                self.linemod_dir, self.pvnet_linemod_dir, obj_name=self.obj_name
            )
            data["RT"] = pose_transformer.orig_pose_to_blender_pose(pose).astype(
                np.float32
            )
            data["cls_typ"] = self.obj_name
            data["rnd_typ"] = "real"
            data["corners"] = projector.project(
                modeldb.get_corners_3d(self.obj_name), data["RT"], "linemod"
            )
            data["farthest"] = projector.project(
                modeldb.get_farthest_3d(self.obj_name), data["RT"], "linemod"
            )
            for num in [4, 12, 16, 20]:
                data["farthest{}".format(num)] = projector.project(
                    modeldb.get_farthest_3d(self.obj_name, num), data["RT"], "linemod",
                )
            data["center"] = projector.project(
                modeldb.get_centers_3d(self.obj_name)[None, :], data["RT"], "linemod"
            )
            data["small_bbox"] = projector.project(
                modeldb.get_small_bbox(self.obj_name), data["RT"], "linemod"
            )
            axis_direct = np.concatenate([np.identity(3), np.zeros([3, 1])], 1).astype(
                np.float32
            )
            data["van_pts"] = projector.project_h(axis_direct, data["RT"], "linemod")
            database.append(data)

        save_pickle(database, self.real_pkl)
        return database

    def collect_train_val_test_info(self):
        with open(os.path.join(self.pvnet_linemod_dir, self.test_fn), "r") as f:
            test_fns = [line.strip().split("/")[-1] for line in f.readlines()]

        with open(os.path.join(self.pvnet_linemod_dir, self.train_fn), "r") as f:
            train_fns = [line.strip().split("/")[-1] for line in f.readlines()]

        with open(os.path.join(self.pvnet_linemod_dir, self.val_fn), "r") as f:
            val_fns = [line.strip().split("/")[-1] for line in f.readlines()]

        for data in self.real_set:
            if data["rgb_pth"].split("/")[-1] in test_fns:
                if data["rgb_pth"].split("/")[-1] in val_fns:
                    self.val_real_set.append(data)
                else:
                    self.test_real_set.append(data)

            if data["rgb_pth"].split("/")[-1] in train_fns:
                self.train_real_set.append(data)

    def collect_fuse_info(self):
        database = []
        modeldb = PVNetLineModModelDB()
        projector = Projector()
        for k in range(self.fuse_num):
            data = dict()
            data["rgb_pth"] = os.path.join(self.fuse_dir, "{}_rgb.jpg".format(k))
            data["dpt_pth"] = os.path.join(self.fuse_dir, "{}_mask.png".format(k))

            # if too few foreground pts then continue
            mask = imread(os.path.join(self.pvnet_linemod_dir, data["dpt_pth"]))
            if np.sum(mask == (cfg.linemod_obj_names.index(self.obj_name) + 1)) < 400:
                continue

            data["cls_typ"] = self.obj_name
            data["rnd_typ"] = "fuse"
            begins, poses = read_pickle(
                os.path.join(self.pvnet_linemod_dir, self.fuse_dir, "{}_info.pkl".format(k))
            )
            data["RT"] = poses[self.obj_idx]
            K = projector.intrinsic_matrix["linemod"].copy()
            K[0, 2] += begins[self.obj_idx, 1]
            K[1, 2] += begins[self.obj_idx, 0]
            data["K"] = K
            data["corners"] = projector.project_K(
                modeldb.get_corners_3d(self.obj_name), data["RT"], K
            )
            data["center"] = projector.project_K(
                modeldb.get_centers_3d(self.obj_name), data["RT"], K
            )
            data["farthest"] = projector.project_K(
                modeldb.get_farthest_3d(self.obj_name), data["RT"], K
            )
            for num in [4, 12, 16, 20]:
                data["farthest{}".format(num)] = projector.project_K(
                    modeldb.get_farthest_3d(self.obj_name, num), data["RT"], K
                )
            data["small_bbox"] = projector.project_K(
                modeldb.get_small_bbox(self.obj_name), data["RT"], K
            )
            database.append(data)

        save_pickle(database, self.fuse_pkl)
        return database


if __name__ == "__main__":
    obj_name = "ape"

    db = PVNetLineModImageDB(obj_name=obj_name)
    print(db)
