import os
import sys
sys.path.append('..')
sys.path.append('../../')

#import bpy
import numpy as np
from transforms3d.euler import mat2euler
#from mathutils import Matrix

from src.config.config import cfg
from src.datasets.LineMod.LineModDB import LineModDB
from src.utils.base_utils import read_ply_model


class ModelAligner(object):
    """
    オブジェクトの姿勢をオリジナルの LineMod のものと Blender 上のものに変換するクラス
    """
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {
        # 'cat': np.array([-0.00577495, -0.01259045, -0.04062323])
    }
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

    def __init__(self,
                              obj_name: str,
                              pvnet_linemod_dir: str,
                              linemod_dir: str):
        """
        ModelAligner の初期化関数

        Args:
            obj_name(str): LineMod データセットに含まれるオブジェクト名
            pvnet_linemod_dir(str): PVNet_LineMod データセットのディレクトリパス
            linemod_dir(str): LineMod データセットのディレクトリパス
        """
        self.obj_name = obj_name
        self.linemod_dir = linemod_dir
        self.LineModDB = LineModDB(linemod_dir = self.linemod_dir,
                                                                       obj_name=self.obj_name)
        self.blender_model_path = os.path.join(pvnet_linemod_dir,
                                                                                           '{}/{}.ply'.format(obj_name, obj_name))
        self.R_p2w, self.t_p2w, self.s_p2w = self.setup_p2w_transform()

    @staticmethod
    def setup_p2w_transform():
        transform1 = np.array([[0.161513626575, -0.827108919621, 0.538334608078, -0.245206743479],
                                                        [-0.986692547798, -0.124983474612, 0.104004733264, -0.050683632493],
                                                        [-0.018740313128, -0.547968924046, -0.836288750172, 0.387638419867]])
        transform2 = np.array([[0.976471602917, 0.201606079936, -0.076541729271, -0.000718327821],
                                                        [-0.196746662259, 0.978194475174, 0.066531419754, 0.000077120210],
                                                        [0.088285841048, -0.049906700850, 0.994844079018, -0.001409600372]])

        R1 = transform1[:, :3]
        t1 = transform1[:, 3]
        R2 = transform2[:, :3]
        t2 = transform2[:, 3]

        # printer system to world system
        t_p2w = np.dot(R2, t1) + t2
        R_p2w = np.dot(R2, R1)
        s_p2w = 0.85
        return R_p2w, t_p2w, s_p2w

    def pose_p2w(self, RT):
        """
        ある姿勢行列を別の姿勢行列に変換する関数

        Arg:
            RT(np.matrix): 変換前の姿勢行列

        Return:
            (np.matrix): 変換後の姿勢行列
        """
        t,R = RT[:,3],RT[:,:3]
        R_w2c = np.dot(R, self.R_p2w.T)
        t_w2c = -np.dot(R_w2c, self.t_p2w) + self.s_p2w * t
        return np.concatenate([R_w2c, t_w2c[:, None]], 1)

    def get_translation_transform(self) -> np.ndarray:
        """
        Blender 上で作成したオブジェクトの3Dモデルとオリジナルの LineMod オブジェクトの3Dモデルを読み出し，この2つのオブジェクトの相対的並進配列を返す関数

        Return:
            (np.ndarray): 2つのオブジェクトの相対的並進配列
        """
        if self.obj_name in self.translation_transforms:
            return self.translation_transforms[self.obj_name]

        blender_model = read_ply_model(self.blender_model_path)
        orig_model = self.LineModDB.load_ply_model()
        blender_model = np.dot(blender_model, self.rotation_transform.T)
        translation_transform = np.mean(orig_model, axis=0) - np.mean(blender_model, axis=0)
        self.translation_transforms[self.obj_name] = translation_transform

        return translation_transform



class PoseTransformer(object):
    rotation_transform = np.array([[1., 0., 0.],
                                                                     [0., -1., 0.],
                                                                     [0., 0., -1.]])
    translation_transforms = {}
    obj_name_to_number = {
        'ape': '001',
        'can': '004',
        'cat': '005',
        'driller': '006',
        'duck': '007',
        'eggbox': '008',
        'glue': '009',
        'holepuncher': '010'
    }
    blender_models={}

    def __init__(self,
                              obj_name: str,
                              pvnet_linemod_dir: str,
                              linemod_dir: str):
        """
        PoseTransformer の初期化関数

        Args:
            obj_name(str): LineMod データセットに含まれるオブジェクト名
            pvnet_linemod_dir(str): PVNet_LineMod データセットのディレクトリパス
            linemod_dir(str): LineMod データセットのディレクトリパス
        """
        self.obj_name = obj_name
        self.blender_model_path = os.path.join(pvnet_linemod_dir, '{}/{}.ply'.format(obj_name, obj_name))
        self.orig_model_path = os.path.join(linemod_dir, '{}/mesh.ply'.format(obj_name))
        self.model_aligner = ModelAligner(obj_name,
                                                                                pvnet_linemod_dir,
                                                                                linemod_dir)


    def orig_pose_to_blender_pose(self, pose: np.matrix) -> np.matrix:
        """
        オリジナルの LineMod データセットのオブジェクトの姿勢から Blender 上 でのオブジェクトの姿勢に変換するための関数

        Arg:
            pose(np.matrix): オリジナルの LineMod データセットのオブジェクトの姿勢情報([R|t] 行列)

        Return:
            (np.matrix): Blender 上 でのオブジェクトの姿勢情報([R|t] 行列)
        """
        rot, tra = pose[:, :3], pose[:, 3]
        tra = tra + np.dot(rot, self.model_aligner.get_translation_transform())
        rot = np.dot(rot, self.rotation_transform)
        return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

    @staticmethod
    def blender_pose_to_blender_euler(pose: np.matrix):
        """
        Blender 上のオブジェクトの姿勢情報を オイラー角表現に変換するための関数

        Args:
            pose (np.matrix): Blender 上 でのオブジェクトの姿勢情報([R|t] 行列)

        Returns:
            [type]: [description]
        """
        euler = [r / np.pi * 180 for r in mat2euler(pose, axes='szxz')]
        euler[0] = -(euler[0] + 90) % 360
        euler[1] = euler[1] - 90
        return np.array(euler)


    def orig_pose_to_blender_euler(self, pose):
        blender_pose = self.orig_pose_to_blender_pose(pose)
        return self.blender_pose_to_blender_euler(blender_pose)

# we could also define the camera matrix
# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_calibration_matrix_K_from_blender(camera):
    """Blender上のカメラ行列を計算する関数

    Args:
        camera ([type]): [description]

    Returns:
        [type]: [description]
    """
    f_in_mm = camera.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera.sensor_width
    sensor_height_in_mm = camera.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camera.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = Matrix(((alpha_u, skew, u_0),
                (0, alpha_v, v_0),
                (0, 0, 1)))

    return K


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(camera):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Use matrix_world instead to account for all constraints
    location, rotation = camera.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv * R_world2bcam
    T_world2cv = R_bcam2cv * T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((R_world2cv[0][:] + (T_world2cv[0],),
                 R_world2cv[1][:] + (T_world2cv[1],),
                 R_world2cv[2][:] + (T_world2cv[2],)))
    return RT


def get_3x4_P_matrix_from_blender(camera):
    K = get_calibration_matrix_K_from_blender(camera.data)
    RT = get_3x4_RT_matrix_from_blender(camera)
    return K*RT


def get_K_P_from_blender(camera):
    K = get_calibration_matrix_K_from_blender(camera.data)
    RT = get_3x4_RT_matrix_from_blender(camera)
    return {"K": np.asarray(K, dtype=np.float32), "RT": np.asarray(RT, dtype=np.float32)}
