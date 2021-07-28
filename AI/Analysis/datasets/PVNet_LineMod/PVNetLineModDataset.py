import json
import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from datasets.augmentation import (
    blur_image,
    mask_out_instance,
    rotate_instance,
    crop_resize_instance_v2,
    crop_resize_instance_v1,
    crop_or_padding_to_fixed_size,
    flip,
)
from datasets.PVNet_LineMod.PVNetLineModModelDB import PVNetLineModModelDB
from src.config.config import cfg
from src.utils.base_utils import read_rgb_np, read_mask_np


def compute_vertex_hcoords(mask, hcoords, use_motion=False):
    h, w = mask.shape
    m = hcoords.shape[0]
    xy = np.argwhere(mask == 1)[:, [1, 0]]
    vertex = xy[:, None, :] * hcoords[None, :, 2:]
    vertex = hcoords[None, :, :2] - vertex
    if not use_motion:
        norm = np.linalg.norm(vertex, axis=2, keepdims=True)
        norm[norm < 1e-3] += 1e-3
        vertex = vertex / norm

    vertex_out = np.zeros([h, w, m, 2], np.float32)
    vertex_out[xy[:, 1], xy[:, 0]] = vertex
    return np.reshape(vertex_out, [h, w, m * 2])


class VotingType:
    BB8 = 0
    BB8C = 1
    BB8S = 2
    VanPts = 3
    Farthest = 5
    Farthest4 = 6
    Farthest12 = 7
    Farthest16 = 8
    Farthest20 = 9

    @staticmethod
    def get_data_pts_2d(vote_type, data):
        if vote_type == VotingType.BB8:
            cor = data["corners"].copy()  # note the copy here!!!
            hcoords = np.concatenate([cor, np.ones([8, 1], np.float32)], 1)  # [8,3]
        elif vote_type == VotingType.BB8C:
            cor = data["corners"].copy()
            cen = data["center"].copy()
            hcoords = np.concatenate([cor, cen], 0)
            hcoords = np.concatenate([hcoords, np.ones([9, 1], np.float32)], 1)
        elif vote_type == VotingType.BB8S:
            cor = data["small_bbox"].copy()
            cen = data["center"].copy()
            hcoords = np.concatenate([cor, cen], 0)
            hcoords = np.concatenate([hcoords, np.ones([9, 1], np.float32)], 1)
        elif vote_type == VotingType.VanPts:
            cen = data["center"].copy()
            van = data["van_pts"].copy()
            hcoords = np.concatenate([cen, np.ones([1, 1], np.float32)], 1)
            hcoords = np.concatenate([van, hcoords], 0)
        elif vote_type == VotingType.Farthest:
            cen = data["center"].copy()
            far = data["farthest"].copy()
            hcoords = np.concatenate([far, cen], 0)
            hcoords = np.concatenate(
                [hcoords, np.ones([hcoords.shape[0], 1], np.float32)], 1
            )
        elif vote_type == VotingType.Farthest4:
            cen = data["center"].copy()
            far = data["farthest4"].copy()
            hcoords = np.concatenate([far, cen], 0)
            hcoords = np.concatenate(
                [hcoords, np.ones([hcoords.shape[0], 1], np.float32)], 1
            )
        elif vote_type == VotingType.Farthest12:
            cen = data["center"].copy()
            far = data["farthest12"].copy()
            hcoords = np.concatenate([far, cen], 0)
            hcoords = np.concatenate(
                [hcoords, np.ones([hcoords.shape[0], 1], np.float32)], 1
            )
        elif vote_type == VotingType.Farthest16:
            cen = data["center"].copy()
            far = data["farthest16"].copy()
            hcoords = np.concatenate([far, cen], 0)
            hcoords = np.concatenate(
                [hcoords, np.ones([hcoords.shape[0], 1], np.float32)], 1
            )
        elif vote_type == VotingType.Farthest20:
            cen = data["center"].copy()
            far = data["farthest20"].copy()
            hcoords = np.concatenate([far, cen], 0)
            hcoords = np.concatenate(
                [hcoords, np.ones([hcoords.shape[0], 1], np.float32)], 1
            )

        return hcoords

    @staticmethod
    def get_pts_3d(vote_type, class_type):
        linemod_db = PVNetLineModModelDB()
        if vote_type == VotingType.BB8C:
            points_3d = linemod_db.get_corners_3d(class_type)
            points_3d = np.concatenate(
                [points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0
            )
        elif vote_type == VotingType.BB8S:
            points_3d = linemod_db.get_small_bbox(class_type)
            points_3d = np.concatenate(
                [points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0
            )
        elif vote_type == VotingType.Farthest:
            points_3d = linemod_db.get_farthest_3d(class_type)
            points_3d = np.concatenate(
                [points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0
            )
        elif vote_type == VotingType.Farthest4:
            points_3d = linemod_db.get_farthest_3d(class_type, 4)
            points_3d = np.concatenate(
                [points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0
            )
        elif vote_type == VotingType.Farthest12:
            points_3d = linemod_db.get_farthest_3d(class_type, 12)
            points_3d = np.concatenate(
                [points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0
            )
        elif vote_type == VotingType.Farthest16:
            points_3d = linemod_db.get_farthest_3d(class_type, 16)
            points_3d = np.concatenate(
                [points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0
            )
        elif vote_type == VotingType.Farthest20:
            points_3d = linemod_db.get_farthest_3d(class_type, 20)
            points_3d = np.concatenate(
                [points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0
            )
        else:  # BB8
            points_3d = linemod_db.get_corners_3d(class_type)

        return points_3d


with open(os.path.join(cfg.CONFIG_DIR, "default_linemod_cfg.json"), "r",) as f:
    default_aug_cfg = json.load(f)


class PVNetLineModDatasetRealAug(Dataset):
    def __init__(
        self,
        imagedb,
        data_prefix: str = cfg.PVNET_LINEMOD_DIR,
        vote_type=VotingType.BB8,
        augment: bool = False,
        cfg=default_aug_cfg,
        background_mask_out: bool = False,
        use_intrinsic: bool = False,
        use_motion: bool = False,
    ):
        super(PVNetLineModDatasetRealAug, self).__init__()
        self.imagedb = imagedb
        self.augment = augment
        self.background_mask_out = background_mask_out
        self.use_intrinsic = use_intrinsic
        self.use_motion = use_motion
        self.cfg = cfg

        self.img_transforms = transforms.Compose(
            [
                transforms.ColorJitter(
                    self.cfg["brightness"],
                    self.cfg["contrast"],
                    self.cfg["saturation"],
                    self.cfg["hue"],
                ),
                transforms.ToTensor(),  # if image.dtype is np.uint8, then it will be divided by 255
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.test_img_transforms = transforms.Compose(
            [
                transforms.ToTensor(),  # if image.dtype is np.uint8, then it will be divided by 255
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.vote_type = vote_type
        self.data_prefix = data_prefix

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple

        rgb_path = os.path.join(self.data_prefix, self.imagedb[index]["rgb_pth"])
        mask_path = os.path.join(self.data_prefix, self.imagedb[index]["dpt_pth"])

        pose = self.imagedb[index]["RT"].copy()
        rgb = read_rgb_np(rgb_path)
        mask = read_mask_np(mask_path)
        if self.imagedb[index]["rnd_typ"] == "real" and len(mask.shape) == 3:
            mask = np.sum(mask, 2) > 0
            mask = np.asarray(mask, np.int32)

        if self.imagedb[index]["rnd_typ"] == "fuse":
            mask = np.asarray(
                mask
                == (cfg.linemod_obj_names.index(self.imagedb[index]["cls_typ"]) + 1),
                np.int32,
            )

        hcoords = VotingType.get_data_pts_2d(self.vote_type, self.imagedb[index])

        if self.use_intrinsic:
            K = torch.tensor(self.imagedb[index]["K"].astype(np.float32))

        if self.augment:
            rgb, mask, hcoords = self.augmentation(rgb, mask, hcoords, height, width)

        ver = compute_vertex_hcoords(mask, hcoords, self.use_motion)
        ver = torch.tensor(ver, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(np.ascontiguousarray(mask), dtype=torch.int64)
        ver_weight = mask.unsqueeze(0).float()

        if self.augment:  # and self.imagedb[index]['rnd_typ']!='real':
            # if not real and do augmentation then jitter color
            if self.cfg["blur"] and np.random.random() < 0.5:
                blur_image(rgb, np.random.choice([3, 5, 7, 9]))
            if self.cfg["jitter"]:
                rgb = self.img_transforms(
                    Image.fromarray(np.ascontiguousarray(rgb, np.uint8))
                )
            else:
                rgb = self.test_img_transforms(
                    Image.fromarray(np.ascontiguousarray(rgb, np.uint8))
                )
            if self.cfg["use_mask_out"] and np.random.random() < 0.1:
                rgb *= (mask[None, :, :]).float()
        else:
            rgb = self.test_img_transforms(
                Image.fromarray(np.ascontiguousarray(rgb, np.uint8))
            )

        if (
            self.imagedb[index]["rnd_typ"] == "fuse"
            and self.cfg["ignore_fuse_ms_vertex"]
        ):
            ver_weight *= 0.0

        pose = torch.tensor(pose.astype(np.float32))
        hcoords = torch.tensor(hcoords.astype(np.float32))
        if self.use_intrinsic:
            return rgb, mask, ver, ver_weight, pose, hcoords, K
        else:
            return rgb, mask, ver, ver_weight, pose, hcoords

    def __len__(self):
        return len(self.imagedb)

    def augmentation(self, img: np.ndarray, mask: np.ndarray, hcoords, height, width):
        foreground = np.sum(mask)
        # randomly mask out to add occlusion
        if self.cfg["mask"] and np.random.random() < 0.5:
            img, mask = mask_out_instance(
                img, mask, self.cfg["min_mask"], self.cfg["max_mask"]
            )

        if foreground > 0:
            # randomly rotate around the center of the instance
            if self.cfg["rotation"]:
                img, mask, hcoords = rotate_instance(
                    img, mask, hcoords, self.cfg["rot_ang_min"], self.cfg["rot_ang_max"]
                )

            # randomly crop and resize
            if self.cfg["crop"]:
                if not self.cfg["use_old"]:
                    # 1. Under 80% probability, we resize the image, which will ensure the size of instance is [hmin,hmax][wmin,wmax]
                    #    otherwise, keep the image unchanged
                    # 2. crop or padding the image to a fixed size
                    img, mask, hcoords = crop_resize_instance_v2(
                        img,
                        mask,
                        hcoords,
                        height,
                        width,
                        self.cfg["overlap_ratio"],
                        self.cfg["resize_hmin"],
                        self.cfg["resize_hmax"],
                        self.cfg["resize_wmin"],
                        self.cfg["resize_wmax"],
                    )
                else:
                    # 1. firstly crop a region which is [scale_min,scale_max]*[height,width], which ensures that
                    #    the area of the intersection between the cropped region and the instance region is at least
                    #    overlap_ratio**2 of instance region.
                    # 2. if the region is larger than original image, then padding 0
                    # 3. then resize the cropped image to [height, width] (bilinear for image, nearest for mask)
                    img, mask, hcoords = crop_resize_instance_v1(
                        img,
                        mask,
                        hcoords,
                        height,
                        width,
                        self.cfg["overlap_ratio"],
                        self.cfg["resize_ratio_min"],
                        self.cfg["resize_ratio_max"],
                    )
        else:
            img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)

        # randomly flip
        if self.cfg["flip"] and np.random.random() < 0.5:
            img, mask, hcoords = flip(img, mask, hcoords)

        return img, mask, hcoords


if __name__ == "__main__":
    from datasets.PVNet_LineMod.PVNetLineModImageDB import PVNetLineModImageDB

    linemod_cls = "ape"
    img_db = PVNetLineModImageDB(linemod_cls, has_render_set=True, has_fuse_set=True)

    test_set = PVNetLineModDatasetRealAug(img_db, cfg.PVNET_LINEMOD_DIR)
    print(test_set)
