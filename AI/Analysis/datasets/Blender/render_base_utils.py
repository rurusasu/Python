import os, sys
from glob import glob

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

#import bpy

# import cv2
import json
import numpy as np
#from mathutils import Matrix
from PIL import Image
from scipy import stats
from skimage.io import imread
#from transforms3d.euler import mat2euler
#from transforms3d.quaternions import mat2quat

from src.config.config import cfg
from datasets.LineMod.LineModDB import LineModDB, read_pose
from src.utils.base_utils import read_pickle, read_ply_model, save_pickle


class ModelAligner(object):
    """
    3Dオブジェクトの姿勢をオリジナルの LineMod のものと Blender 上のものに変換するクラス
    """

    # X軸 -180度回転する場合の回転行列
    rotation_transform = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    translation_transforms = {
        # 'cat': np.array([-0.00577495, -0.01259045, -0.04062323])
    }
    intrinsic_matrix = {
        "linemod": np.array(
            [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
        ),
        # 'blender': np.array([[280.0, 0.0, 128.0],
        #                      [0.0, 280.0, 128.0],
        #                      [0.0, 0.0, 1.0]]),
        "blender": np.array(
            [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]]
        ),
    }

    def __init__(self, linemod_dir: str, pvnet_linemod_dir: str, obj_name: str):
        """
        ModelAligner の初期化関数

        Args:
            linemod_dir(str): LineMod データセットのディレクトリパス
            pvnet_linemod_dir(str): PVNet_LineMod データセットのディレクトリパス
            obj_name(str): LineMod データセットに含まれるオブジェクト名
        """
        self.obj_name = obj_name
        self.linemod_dir = linemod_dir
        self.LineModDB = LineModDB(linemod_dir=self.linemod_dir, obj_name=self.obj_name)
        self.blender_model_path = os.path.join(
            pvnet_linemod_dir, "{}/{}.ply".format(obj_name, obj_name)
        )
        self.R_p2w, self.t_p2w, self.s_p2w = self.setup_p2w_transform()

    @staticmethod
    def setup_p2w_transform():
        transform1 = np.array(
            [
                [0.161513626575, -0.827108919621, 0.538334608078, -0.245206743479],
                [-0.986692547798, -0.124983474612, 0.104004733264, -0.050683632493],
                [-0.018740313128, -0.547968924046, -0.836288750172, 0.387638419867],
            ]
        )
        transform2 = np.array(
            [
                [0.976471602917, 0.201606079936, -0.076541729271, -0.000718327821],
                [-0.196746662259, 0.978194475174, 0.066531419754, 0.000077120210],
                [0.088285841048, -0.049906700850, 0.994844079018, -0.001409600372],
            ]
        )

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
        t, R = RT[:, 3], RT[:, :3]
        R_w2c = np.dot(R, self.R_p2w.T)
        t_w2c = -np.dot(R_w2c, self.t_p2w) + self.s_p2w * t
        return np.concatenate([R_w2c, t_w2c[:, None]], 1)

    def get_translation_transform(self) -> np.ndarray:
        """
        Blender 上で作成したオブジェクトの3Dモデルとオリジナルの LineMod オブジェクトの3Dモデルを読み出し，後者を前者の位置へ移動させるための並進ベクトルを計算する関数
        Return:
            translation_transform(np.matrix): 2つのオブジェクトの相対的並進ベクトル(3x1)
        """
        if self.obj_name in self.translation_transforms:
            return self.translation_transforms[self.obj_name]

        blender_model = read_ply_model(self.blender_model_path)
        orig_model = self.LineModDB.load_ply_model()
        blender_model = np.dot(blender_model, self.rotation_transform.T)
        translation_transform = np.mean(orig_model, axis=0) - np.mean(
            blender_model, axis=0
        )  # 各モデルごとに点群座標全体の平均値を計算し，引き算することで並進ベクトルを求める．
        self.translation_transforms[self.obj_name] = translation_transform

        return translation_transform


class PoseTransformer(object):
    rotation_transform = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    translation_transforms = {}
    obj_name_to_number = {
        "ape": "001",
        "can": "004",
        "cat": "005",
        "driller": "006",
        "duck": "007",
        "eggbox": "008",
        "glue": "009",
        "holepuncher": "010",
    }
    blender_models = {}

    def __init__(self, linemod_dir: str, pvnet_linemod_dir: str, obj_name: str):
        """
        PoseTransformer の初期化関数
        Args:
            linemod_dir(str): LineMod データセットのディレクトリパス
            pvnet_linemod_dir(str): PVNet_LineMod データセットのディレクトリパス
            obj_name(str): LineMod データセットに含まれるオブジェクト名
        """
        self.obj_name = obj_name
        self.blender_model_path = os.path.join(
            pvnet_linemod_dir, "{}/{}.ply".format(obj_name, obj_name)
        )
        self.orig_model_path = os.path.join(linemod_dir, "{}/mesh.ply".format(obj_name))
        self.model_aligner = ModelAligner(
            linemod_dir=linemod_dir,
            pvnet_linemod_dir=pvnet_linemod_dir,
            obj_name=obj_name,
        )

    def orig_pose_to_blender_pose(self, pose: np.matrix) -> np.ndarray:
        """
        オリジナルの LineMod データセットのオブジェクトの姿勢から Blender 上 でのオブジェクトの姿勢に変換するための関数
        Arg:
            pose(np.matrix): オリジナルの LineMod データセットの`.rot`, `.tra` ファイルから読み出したオブジェクトの同次変換行列([R|t] 行列)．
        Return:
            (np.matrix): Blender 上 でのオブジェクトの同次変換行列([R|t] 行列)
        """
        rot, tra = pose[:, :3], pose[:, 3]
        tra = tra + np.dot(rot, self.model_aligner.get_translation_transform())
        rot = np.dot(rot, self.rotation_transform)  # R_orig × X軸回りに-180度回転
        return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

    @staticmethod
    def blender_pose_to_blender_euler(pose: np.matrix):
        """
        Blender 上のオブジェクトの姿勢情報を オイラー角表現に変換するための関数
        Args:
            pose (np.matrix): Blender 上 でのオブジェクトの同次変換行列([R|t] 行列)
        Returns:
            [type]: [description]
        """
        euler = [r / np.pi * 180 for r in mat2euler(pose, axes="szxz")]
        euler[0] = -(euler[0] + 90) % 360
        euler[1] = euler[1] - 90
        return np.array(euler)

    def orig_pose_to_blender_euler(self, pose):
        blender_pose = self.orig_pose_to_blender_pose(pose)
        return self.blender_pose_to_blender_euler(blender_pose)


class DataStatistics(object):
    # world_to_camera_pose = np.array([[-1.19209304e-07,   1.00000000e+00,  -2.98023188e-08, 1.19209304e-07],
    #                                  [-8.94069672e-08,   2.22044605e-16,  -1.00000000e+00, 8.94069672e-08],
    #                                  [-1.00000000e+00,  -8.94069672e-08,   1.19209304e-07, 1.00000000e+00]])
    world_to_camera_pose = np.array(
        [
            [-1.00000024e00, -8.74227979e-08, -5.02429621e-15, 8.74227979e-08],
            [5.02429621e-15, 1.34358856e-07, -1.00000012e00, -1.34358856e-07],
            [8.74227979e-08, -1.00000012e00, 1.34358856e-07, 1.00000012e00],
        ]
    )

    def __init__(self, linemod_dir: str, pvnet_linemod_dir: str, obj_name):
        """
        DataStatistics の初期化関数

        Args:
            linemod_dir(str): LineMod データセットのディレクトリパス
            pvnet_linemod_dir(str): PVNet_LineMod データセットのディレクトリパス
            obj_name(str): LineMod データセットに含まれるオブジェクト名
                dataset_collect_linemod_set_info で作成される `{obj_name}_info.pkl` の保存先のディレクトリパス
        """
        self.obj_name = obj_name
        self.linemod_dir = linemod_dir
        self.pvnet_linemod_dir = pvnet_linemod_dir

        self.mask_path = os.path.join(
            self.pvnet_linemod_dir, "{}/mask/*.png".format(obj_name)
        )
        self.dir_path = os.path.join(self.linemod_dir, "{}/data".format(obj_name))

        dataset_pose_dir_path = os.path.join(cfg.DATA_DIR, "dataset_poses")
        os.system("mkdir -p {}".format(dataset_pose_dir_path))
        self.dataset_poses_path = os.path.join(
            dataset_pose_dir_path, "{}_poses.npy".format(obj_name)
        )
        blender_pose_dir_path = os.path.join(cfg.DATA_DIR, "blender_poses")
        os.system("mkdir -p {}".format(blender_pose_dir_path))
        self.blender_poses_path = os.path.join(
            blender_pose_dir_path, "{}_poses.npy".format(obj_name)
        )
        os.system("mkdir -p {}".format(blender_pose_dir_path))

        self.pose_transformer = PoseTransformer(
            linemod_dir=self.linemod_dir,
            pvnet_linemod_dir=self.pvnet_linemod_dir,
            obj_name=self.obj_name,
        )

        try:
            with open("config.json", mode="rt", encoding="utf-8") as f:
                self.num_samples = json.load(f)["Renderer"]["num_samples"]
        except FileNotFoundError as fe:
            print("config.json ファイルが見つかりません．デフォルト値を使用します．")
            self.num_samples = 10
        except Exception as e:
            print("予期せぬ例外です．")
        # self.num_samples = json.load()

    def get_proper_crop_size(self):
        """
        カテゴリごとのマスク画像におけるマスク部分(非0部分)の縦横の最大最小サイズを求める関数．
        """
        mask_paths = glob(self.mask_path)
        widths = []
        heights = []

        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert("1")
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
        print(
            "min width: {} px, max width: {} px".format(np.min(widths), np.max(widths))
        )
        print(
            "min height: {} px, max height: {} px".format(
                np.min(heights), np.max(heights)
            )
        )

    def get_quat_translation(self, object_to_camera_pose):
        object_to_camera_pose = np.append(object_to_camera_pose, [[0, 0, 0, 1]], axis=0)
        world_to_camera_pose = np.append(
            self.world_to_camera_pose, [[0, 0, 0, 1]], axis=0
        )
        object_to_world_pose = np.dot(
            np.linalg.inv(world_to_camera_pose), object_to_camera_pose
        )
        quat = mat2quat(object_to_world_pose[:3, :3])
        translation = object_to_world_pose[:3, 3]
        return quat, translation

    def get_dataset_poses(self):
        """

        Returns:
            [type]: [description]
        """
        # 既にファイルが存在する場合
        if os.path.exists(self.dataset_poses_path):
            poses = np.load(self.dataset_poses_path)
            return poses[:, :3], poses[:, 3:]

        eulers = []
        translations = []
        train_set = np.loadtxt(
            os.path.join(
                self.pvnet_linemod_dir, "{}/training_range.txt".format(self.obj_name)
            ),
            np.int32,
        )
        for idx in train_set:
            rot_path = os.path.join(self.dir_path, "rot{}.rot".format(idx))
            tra_path = os.path.join(self.dir_path, "tra{}.tra".format(idx))
            pose = read_pose(rot_path, tra_path)
            euler = self.pose_transformer.orig_pose_to_blender_euler(pose)
            eulers.append(euler)
            translations.append(pose[:, 3])

        eulers = np.array(eulers)
        translations = np.array(translations)
        np.save(
            self.dataset_poses_path, np.concatenate([eulers, translations], axis=-1)
        )

        return eulers, translations

    def sample_sphere(self, num_samples):
        """ sample angles from the sphere
        reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
        """
        flat_objects = ["037_scissors", "051_large_clamp", "052_extra_large_clamp"]
        if self.obj_name in flat_objects:
            begin_elevation = 30
        else:
            begin_elevation = 0
        ratio = (begin_elevation + 90) / 180
        num_points = int(num_samples // (1 - ratio))
        phi = (np.sqrt(5) - 1.0) / 2.0
        azimuths = []
        elevations = []
        for n in range(num_points - num_samples, num_points):
            z = 2.0 * n / num_points - 1.0
            azimuths.append(np.rad2deg(2 * np.pi * n * phi % (2 * np.pi)))
            elevations.append(np.rad2deg(np.arcsin(z)))
        return np.array(azimuths), np.array(elevations)

    def sample_poses(self):
        eulers, translations = self.get_dataset_poses()
        num_samples = self.num_samples
        azimuths, elevations = self.sample_sphere(num_samples)
        euler_sampler = stats.gaussian_kde(eulers.T)
        eulers = euler_sampler.resample(num_samples).T
        eulers[:, 0] = azimuths
        eulers[:, 1] = elevations
        translation_sampler = stats.gaussian_kde(translations.T)
        translations = translation_sampler.resample(num_samples).T
        np.save(
            self.blender_poses_path, np.concatenate([eulers, translations], axis=-1)
        )
        return np.concatenate([eulers, translations], axis=-1)



# ----------------- #
# カメラの設定 #
# ----------------- #
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

    if camera.sensor_fit == "VERTICAL":
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

    K = Matrix(((alpha_u, skew, u_0), (0, alpha_v, v_0), (0, 0, 1)))

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
    # カメラオブジェクトをアクティブにする．
    bpy.context.view_layer.objects.active = camera
    # bcam stands for blender camera
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    # Use matrix_world instead to account for all constraints
    # location, rotation = camera.matrix_world.decompose()[0:2]
    location, rotation = bpy.context.object.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],),
        )
    )

    return RT


def get_3x4_P_matrix_from_blender(camera):
    K = get_calibration_matrix_K_from_blender(camera.data)
    RT = get_3x4_RT_matrix_from_blender(camera)
    return K * RT


def get_K_P_from_blender(camera):
    K = get_calibration_matrix_K_from_blender(camera.data)
    RT = get_3x4_RT_matrix_from_blender(camera)
    return {
        "K": np.asarray(K, dtype=np.float32),
        "RT": np.asarray(RT, dtype=np.float32),
    }


if __name__ == "__main__":
    pvnet_linemod_dir = cfg.PVNET_LINEMOD_DIR
    linemod_dir = cfg.LINEMOD_DIR
    obj_name = "ape"

    rotation_transform = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

    # ------------------- #
    # ModelAligner  #
    # ------------------- #
    # ----------------------- #
    # インスタンス生成 #
    # ----------------------- #
    # mdl_aln = ModelAligner(obj_name = obj_name,
    #                                                    pvnet_linemod_dir = pvnet_linemod_dir,
    #                                                    linemod_dir = linemod_dir)

    # ------------------------------------------ #
    # get_translation_transform Test #
    # ------------------------------------------ #
    # t_tfm = mdl_aln.get_translation_transform()
    # print(t_tfm)

    #
    blender_model_path = os.path.join(
        pvnet_linemod_dir, "{}/{}.ply".format(obj_name, obj_name)
    )
    db = LineModDB(linemod_dir=linemod_dir, obj_name=obj_name)

    blender_model = read_ply_model(blender_model_path)
    orig_model = db.load_ply_model()
    blender_model = np.dot(blender_model, rotation_transform.T)  # X軸回りに-180度回転
    # translation_transform = np.mean(orig_model, axis=0) - np.mean(blender_model, axis=0)

    # ----------------------- #
    # PoseTransformer #
    # ----------------------- #
    from datasets.LineMod.LineModDB import read_pose

    # 姿勢
    rt_dir = os.path.join(linemod_dir, obj_name, "data")
    rot_path = os.path.join(rt_dir, "rot{}.rot".format(1))
    tra_path = os.path.join(rt_dir, "tra{}.tra".format(1))
    pose = read_pose(rot_path, tra_path)
    # ----------------------- #
    # インスタンス生成 #
    # ----------------------- #
    pos_tra = PoseTransformer(
        linemod_dir=linemod_dir, pvnet_linemod_dir=pvnet_linemod_dir, obj_name=obj_name
    )
    # -------------------------------------------- #
    # orig_pose_to_blender_pose Test #
    # -------------------------------------------- #
    blender_pos = pos_tra.orig_pose_to_blender_pose(pose)
    print(blender_pos)

    transform_orig_model = np.dot(orig_model, blender_pos)

    # ---------------- #
    # 3D プロット #
    # ---------------- #
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    blender_ply_X = blender_model[:, 0]
    blender_ply_Y = blender_model[:, 1]
    blender_ply_Z = blender_model[:, 2]

    orig_ply_X = orig_model[:, 0]
    orig_ply_Y = orig_model[:, 1]
    orig_ply_Z = orig_model[:, 2]

    transform_orig_ply_X = transform_orig_model[:, 0]
    transform_orig_ply_Y = transform_orig_model[:, 1]
    transform_orig_ply_Z = transform_orig_model[:, 2]

    # グラフの枠を作成
    fig = plt.figure()
    ax = Axes3D(fig)

    # X,Y,Z軸にラベルを設定
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # .scatterで描画
    ax.scatter(
        blender_ply_X,
        blender_ply_Y,
        blender_ply_Z,
        s=5,
        c="plum",
        marker="o",
        alpha=0.1,
        label="blender_model",
    )

    ax.scatter(
        orig_ply_X,
        orig_ply_Y,
        orig_ply_Z,
        s=5,
        c="b",
        marker="o",
        alpha=0.1,
        label="orig_model",
    )

    ax.scatter(
        transform_orig_ply_X,
        transform_orig_ply_Y,
        transform_orig_ply_Z,
        s=5,
        c="g",
        marker="o",
        alpha=0.1,
        label="transform_orig_model",
    )

    # 凡例を表示
    ax.legend()
    # 最後に.show()を書いてグラフ表示
    plt.show()
