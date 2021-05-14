import os, sys
sys.path.append('..')

import glob
import struct

import Imath
import numpy as np
import OpenEXR
import scipy.io as sio
from multiprocessing.dummy import Pool
from PIL import Image
from scipy import stats
#from transforms3d.quaternions import mat2quat

from src.config.config import cfg
from src.utils.base_utils import PoseTransformer, read_pose, read_pickle, save_pickle


class DataStatistics(object):
    # world_to_camera_pose = np.array([[-1.19209304e-07,   1.00000000e+00,  -2.98023188e-08, 1.19209304e-07],
    #                                  [-8.94069672e-08,   2.22044605e-16,  -1.00000000e+00, 8.94069672e-08],
    #                                  [-1.00000000e+00,  -8.94069672e-08,   1.19209304e-07, 1.00000000e+00]])
    world_to_camera_pose = np.array([[-1.00000024e+00,  -8.74227979e-08,  -5.02429621e-15, 8.74227979e-08],
                                                                               [5.02429621e-15,   1.34358856e-07,  -1.00000012e+00, -1.34358856e-07],
                                                                               [8.74227979e-08,  -1.00000012e+00,   1.34358856e-07, 1.00000012e+00]])

    def __init__(self, object_name):
        self.object_name = object_name
        self.linemod_dir = cfg.LINEMOD_DIR
        self.pvnet_linemod_dir = cfg.PVNET_LINEMOD_DIR
        self.parent_dir_path = cfg.DATA_DIR

        self.mask_path = os.path.join(self.pvnet_linemod_dir,'{}/mask/*.png'.format(object_name))
        self.dir_path = os.path.join(self.linemod_dir,'{}/data'.format(object_name))

        dataset_pose_dir_path = os.path.join(self.parent_dir_path, 'dataset_poses')
        os.system('mkdir -p {}'.format(dataset_pose_dir_path))
        self.dataset_poses_path = os.path.join(dataset_pose_dir_path, '{}_poses.npy'.format(object_name))
        blender_pose_dir_path = os.path.join(self.parent_dir_path, 'blender_poses')
        os.system('mkdir -p {}'.format(blender_pose_dir_path))
        self.blender_poses_path = os.path.join(blender_pose_dir_path, '{}_poses.npy'.format(object_name))
        os.system('mkdir -p {}'.format(blender_pose_dir_path))

        self.pose_transformer = PoseTransformer(object_name)

    def get_proper_crop_size(self):
        mask_paths = glob.glob(self.mask_path)
        widths = []
        heights = []

        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert('1')
            mask = np.array(mask).astype(np.int32)
            row_col = np.argwhere(mask == 1)
            min_row, max_row = np.min(row_col[:, 0]), np.max(row_col[:, 0])
            min_col, max_col = np.min(row_col[:, 1]), np.max(row_col[:, 1])
            width = max_col - min_col
            height = max_row - min_row
            widths.append(width)
            heights.append(height)

        widths = np.array(widths)
        heights = np.array(heights)
        print('min width: {}, max width: {}'.format(np.min(widths), np.max(widths)))
        print('min height: {}, max height: {}'.format(np.min(heights), np.max(heights)))

    def get_quat_translation(self, object_to_camera_pose):
        object_to_camera_pose = np.append(object_to_camera_pose, [[0, 0, 0, 1]], axis=0)
        world_to_camera_pose = np.append(self.world_to_camera_pose, [[0, 0, 0, 1]], axis=0)
        object_to_world_pose = np.dot(np.linalg.inv(world_to_camera_pose), object_to_camera_pose)
        quat = mat2quat(object_to_world_pose[:3, :3])
        translation = object_to_world_pose[:3, 3]
        return quat, translation

    def get_dataset_poses(self):
        if os.path.exists(self.dataset_poses_path):
            poses = np.load(self.dataset_poses_path)
            return poses[:, :3], poses[:, 3:]

        eulers = []
        translations = []
        train_set = np.loadtxt(os.path.join(self.pvnet_linemod_dir, '{}/training_range.txt'.format(self.object_name)),np.int32)
        for idx in train_set:
            rot_path = os.path.join(self.dir_path, 'rot{}.rot'.format(idx))
            tra_path = os.path.join(self.dir_path, 'tra{}.tra'.format(idx))
            pose = read_pose(rot_path, tra_path)
            euler = self.pose_transformer.orig_pose_to_blender_euler(pose)
            eulers.append(euler)
            translations.append(pose[:, 3])

        eulers = np.array(eulers)
        translations = np.array(translations)
        np.save(self.dataset_poses_path, np.concatenate([eulers, translations], axis=-1))

        return eulers, translations

    def sample_sphere(self, num_samples):
        """ sample angles from the sphere
        reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
        """
        flat_objects = ['037_scissors', '051_large_clamp', '052_extra_large_clamp']
        if self.object_name in flat_objects:
            begin_elevation = 30
        else:
            begin_elevation = 0
        ratio = (begin_elevation + 90) / 180
        num_points = int(num_samples // (1 - ratio))
        phi = (np.sqrt(5) - 1.0) / 2.
        azimuths = []
        elevations = []
        for n in range(num_points - num_samples, num_points):
            z = 2. * n / num_points - 1.
            azimuths.append(np.rad2deg(2 * np.pi * n * phi % (2 * np.pi)))
            elevations.append(np.rad2deg(np.arcsin(z)))
        return np.array(azimuths), np.array(elevations)

    def sample_poses(self):
        eulers, translations = self.get_dataset_poses()
        num_samples = cfg.NUM_SYN
        azimuths, elevations = self.sample_sphere(num_samples)
        euler_sampler = stats.gaussian_kde(eulers.T)
        eulers = euler_sampler.resample(num_samples).T
        eulers[:, 0] = azimuths
        eulers[:, 1] = elevations
        translation_sampler = stats.gaussian_kde(translations.T)
        translations = translation_sampler.resample(num_samples).T
        np.save(self.blender_poses_path, np.concatenate([eulers, translations], axis=-1))


class YCBDataStatistics(DataStatistics):
    def __init__(self, object_name):
        super(YCBDataStatistics, self).__init__(object_name)
        self.dir_path = os.path.join(self.linemod_dir, '{}/data'.format(object_name))
        self.object_names = np.loadtxt(os.path.join(cfg.YCB, 'image_sets/classes.txt'), dtype=np.str)
        self.object_names = np.insert(self.object_names, 0, 'background')
        self.train_set = np.loadtxt(os.path.join(cfg.YCB, 'image_sets/train.txt'), dtype=np.str)
        self.meta_pattern = os.path.join(cfg.YCB, 'data/{}-meta.mat')
        self.dataset_poses_pattern = os.path.join(self.parent_dir_path, 'dataset_poses/{}_poses.npy')

    def get_dataset_poses(self):
        if os.path.exists(self.dataset_poses_path):
            poses = np.load(self.dataset_poses_pattern.format(self.object_name))
            return poses[:, :3], poses[:, 3:]

        dataset_poses = {}
        for i in self.train_set:
            meta_path = self.meta_pattern.format(i)
            meta = sio.loadmat(meta_path)
            classes = meta['cls_indexes'].ravel()
            poses = meta['poses']
            for idx, cls_idx in enumerate(classes):
                cls_poses = dataset_poses.setdefault(self.object_names[cls_idx], [[], []])
                pose = poses[..., idx]
                euler = self.pose_transformer.blender_pose_to_blender_euler(pose)
                cls_poses[0].append(euler)
                cls_poses[1].append(pose[:, 3])

        for object_name, cls_poses in dataset_poses.items():
            np.save(self.dataset_poses_pattern.format(object_name), np.concatenate(cls_poses, axis=-1))

        cls_poses = dataset_poses[self.object_name]
        eulers = np.array(cls_poses[0])
        translations = np.array(cls_poses[1])

        return eulers, translations


class Renderer(object):
    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                                                  [0., 573.57043, 242.04899],
                                                  [0., 0., 1.]]),
        # 'blender': np.array([[280.0, 0.0, 128.0],
        #                      [0.0, 280.0, 128.0],
        #                      [0.0, 0.0, 1.0]]),
        'blender': np.array([[700.,    0.,  320.],
                                                  [0.,  700.,  240.],
                                                  [0.,    0.,    1.]])
    }

    def __init__(self, object_name: str):
        """
        Renderer の初期化

        Arg:
            object_name (str): 対象とするオブジェクト名
        """
        self.object_name = object_name
        self.linemod_dir = cfg.LINEMOD_DIR
        self.pvnet_linemod_dir = cfg.PVNET_LINEMOD_DIR
        self.parent_dir_path = cfg.DATA_DIR

        self.bg_imgs_path = os.path.join(self.parent_dir_path, 'bg_imgs.npy')
        self.poses_path = os.path.join(self.parent_dir_path, 'blender_poses', '{}_poses.npy').format(object_name)
        self.output_dir_path = os.path.join(self.pvnet_linemod_dir,'renders/{}').format(object_name)
        self.blender_path = cfg.BLENDER_PATH
        self.blank_blend = os.path.join(self.parent_dir_path, 'blank.blend')
        self.py_path = os.path.join(cfg.BLENDER_DIR, 'render_backend.py')
        self.obj_path = os.path.join(self.pvnet_linemod_dir,'{}/{}.ply').format(object_name, object_name)
        self.plane_height_path = os.path.join(self.parent_dir_path, 'plane_height.pkl')

    def get_bg_imgs(self):
        if os.path.exists(self.bg_imgs_path):
            return

        img_paths = glob.glob(os.path.join(cfg.SUN, 'JPEGImages/*'))
        bg_imgs = []

        for img_path in img_paths:
            img = Image.open(img_path)
            row, col = img.size
            if row > 500 and col > 500:
                bg_imgs.append(img_path)

        np.save(self.bg_imgs_path, bg_imgs)

    def project_model(self, model_3d, pose, camera_type):
        camera_model_2d = np.dot(model_3d, pose[:, :3].T) + pose[:, 3]
        camera_model_2d = np.dot(camera_model_2d, self.intrinsic_matrix[camera_type].T)
        return camera_model_2d[:, :2] / camera_model_2d[:, 2:]

    @staticmethod
    def exr_to_png(exr_path):
        depth_path = exr_path.replace('.png0001.exr', '.png')
        exr_image = OpenEXR.InputFile(exr_path)
        dw = exr_image.header()['dataWindow']
        (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        def read_exr(s, width, height):
            mat = np.fromstring(s, dtype=np.float32)
            mat = mat.reshape(height, width)
            return mat

        dmap, _, _ = [read_exr(s, width, height) for s in exr_image.channels('BGR', Imath.PixelType(Imath.PixelType.FLOAT))]
        dmap = Image.fromarray((dmap != 1).astype(np.int32))
        dmap.save(depth_path)
        exr_image.close()
        os.system('rm {}'.format(exr_path))

    def sample_poses(self):
        statistician = DataStatistics(self.object_name)
        statistician.sample_poses()

    def get_plane_height(self):
        if os.path.exists(self.plane_height_path):
            plane_height = read_pickle(self.plane_height_path)
        else:
            plane_height = {}

        if self.object_name in plane_height:
            return plane_height[self.object_name]
        else:
            pose_transformer = PoseTransformer(self.object_name)
            model = pose_transformer.get_blender_model()
            height = np.min(model[:, -1])
            plane_height[self.object_name] = height
            save_pickle(plane_height, self.plane_height_path)
            return height

    def run(self):
        """ Render images
        1. prepare background images
        2. sample poses from the pose distribution of training data
        3. call the blender to render images
        """
        self.get_bg_imgs()
        self.sample_poses()

        if not os.path.exists(self.output_dir_path):
            os.makedirs(self.output_dir_path)

        os.system('{} {} --background --python {} -- --input {} --output_dir {} --bg_imgs {} --poses_path {}'.
                  format(self.blender_path, self.blank_blend, self.py_path, self.obj_path,
                         self.output_dir_path, self.bg_imgs_path, self.poses_path))
        depth_paths = glob.glob(os.path.join(self.output_dir_path, '*.exr'))
        for depth_path in depth_paths:
            self.exr_to_png(depth_path)

    @staticmethod
    def multi_thread_render():
        # objects = ['ape', 'benchvise', 'bowl', 'can', 'cat', 'cup', 'driller', 'duck',
        #            'glue', 'holepuncher', 'iron', 'lamp', 'phone', 'cam', 'eggbox']
        objects = ['lamp', 'phone']

        def render(object_name):
            renderer = Renderer(object_name)
            renderer.run()

        with Pool(processes=2) as pool:
            pool.map(render, objects)


class YCBRenderer(Renderer):
    def __init__(self, object_name):
        super(YCBRenderer, self).__init__(object_name)
        self.output_dir_path = os.path.join(cfg.YCB, 'renders/{}').format(object_name)
        self.blank_blend = os.path.join(self.parent_dir_path, 'blank.blend')
        self.obj_path = os.path.join(cfg.YCB, 'models', object_name, 'textured.obj')
        self.object_names = np.loadtxt(os.path.join(cfg.YCB, 'image_sets/classes.txt'), dtype=np.str)
        self.object_names = np.insert(self.object_names, 0, 'background')

    def sample_poses(self):
        statistician = YCBDataStatistics(self.object_name)
        statistician.sample_poses()

    @staticmethod
    def multi_thread_render():
        objects = ['003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle']

        def render(object_name):
            renderer = YCBRenderer(object_name)
            renderer.run()

        with Pool(processes=2) as pool:
            pool.map(render, objects)


class MultiRenderer(Renderer):
    object_names = ['ape', 'benchvise', 'can', 'cat', 'driller', 'duck', 'glue',
                   'holepuncher', 'iron', 'lamp', 'phone', 'cam', 'eggbox']

    def __init__(self):
        super(MultiRenderer, self).__init__('')
        self.poses_path = os.path.join(self.parent_dir_path, '{}_poses.npy')
        self.output_dir_path = '/home/pengsida/Datasets/LINEMOD/renders/all_objects'

    def sample_poses(self):
        for object_name in self.object_names:
            statistician = DataStatistics(object_name)
            statistician.sample_poses()

    def run(self):
        """ Render images
        1. prepare background images
        2. sample poses from the pose distribution of training data
        3. call the blender to render images
        """
        self.get_bg_imgs()
        self.sample_poses()

        os.system('{} {} --background --python {} -- --input {} --output_dir {} --use_cycles True --bg_imgs {} --poses_path {}'.
                  format(self.blender_path, self.blank_blend, self.py_path, self.obj_path, self.output_dir_path, self.bg_imgs_path, self.poses_path))
        depth_paths = glob.glob(os.path.join(self.output_dir_path, '*.exr'))
        for depth_path in depth_paths:
            self.exr_to_png(depth_path)

