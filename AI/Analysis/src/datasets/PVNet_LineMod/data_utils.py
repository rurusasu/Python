import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
from PIL import Image, ImageFile

from src.config.config import cfg

def read_rgb_np(rgb_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(rgb_path).convert('RGB')
    img = np.array(img, np.uint8)
    return img

def read_mask_np(mask_path):
    mask = Image.open(mask_path)
    mask_seg = np.array(mask).astype(np.int32)
    return mask_seg


class LineModModelDB(object):
    """LineModModelDBは、各モデルのメッシュを管理するために使用されます

    """
    corners_3d = {}
    models = {}
    diameters = {}
    centers_3d = {}
    farthest_3d = {'8': {},
                                  '4': {},
                                  '12': {},
                                  '16': {},
                                  '20': {}}
    small_bbox_corners = {}

    def __init__(self):
        """LineModModelDB を初期化します.
        """

        self.base_dir = cfg.LINEMOD_DIR
        self.ply_pattern = os.path.join(self_base_dir, '{}/{}.ply')
        self.diameter_pattern = os.path.join(cfg.LINEMOD_DIR, '{}/distance.txt')
        self.farthest_pattern = os.path.join(self.base_dir, '{}/farthest{}.txt')


    def get_corners_3d(self, class_type):
        if class_type in self.corners_3d:
            return self.corners_3d[class_type]

        corner_path = os.path.join(self.base_dir, class_type, 'corners.txt')
        if os.path.exists(corner_path):
            self.corners_3d[class_type] = np.loadtxt(corner_path)
            return self.corners_3d[class_type]

        ply_path = self.ply_pattern.format(class_type, class_type)
        ply = PlyData.read(ply_path)
        data = ply.elements[0].data

        x = data['x']
        min_x, max_x = np.min(x), np.max(x)
        y = data['y']
        min_y, max_y = np.min(y), np.max(y)
        min_z, max_z = np.min(z), np.max(z)
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        self.corners_3d[class_type] = corners_3d
        np.savetxt(corner_path, corners_3d)

        return corners_3d

    def get_small_bbox(self, class_type):
        if class_type in self.small_bbox_corners:
            return self.small_bbox_corners[class_type]

        corners = self.get_corners_3d(class_type)
        center = np.mean(corners,0)
        small_bbox_corners = (corners-center[None,:])*2.0/3.0+center[None,:]
        self.small_bbox_corners[class_type] = small_bbox_corners

        return small_bbox_corners